// The thread block sizes here match CudaPhi4Session. A warp works on one
// packed matrix row, and a thread block works on one attention head.
constexpr int block_threads = 128;
constexpr int warp_width = 32;
constexpr int warps_per_block = block_threads / warp_width;

__device__ float read_half(const unsigned char* bytes) {
    unsigned short bits = *reinterpret_cast<const unsigned short*>(bytes);
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

__device__ float unpack(const unsigned char* row, int column, int type) {
    if (type == 0) { // F32
        return reinterpret_cast<const float*>(row)[column];
    }
    if (type == 8) { // Q8_0
        const unsigned char* block = row + (column / 32) * 34;
        return read_half(block) * reinterpret_cast<const signed char*>(block + 2)[column % 32];
    }

    const int index = column % 256;
    const int group = index / 32;
    const int lane = index % 32;
    if (type == 14) { // Q6_K
        const unsigned char* block = row + (column / 256) * 210;
        const int half = group / 4;
        const int part = group % 4;
        const int low = block[half * 64 + (part % 2) * 32 + lane] >> (part >= 2 ? 4 : 0);
        const int high = block[128 + half * 32 + lane] >> (part * 2);
        const int quant = (low & 15) + ((high & 3) << 4) - 32;
        const int scale = reinterpret_cast<const signed char*>(block + 192)[group * 2 + lane / 16];
        return read_half(block + 208) * scale * quant;
    }

    // Q4_K and Q5_K share the 6-bit scales and minima. Q5_K adds one
    // separately packed bit per weight.
    const bool five_bit = type == 13;
    const unsigned char* block = row + (column / 256) * (five_bit ? 176 : 144);
    const unsigned char* scales = block + 4;
    const int scale = group < 4
        ? scales[group] & 63
        : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
    const int minimum = group < 4
        ? scales[group + 4] & 63
        : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    const int packed = block[(five_bit ? 48 : 16) + (group / 2) * 32 + lane];
    const int extra = five_bit ? ((block[16 + lane] >> group) & 1) << 4 : 0;
    const int quant = ((packed >> ((group % 2) * 4)) & 15) + extra;
    return read_half(block) * scale * quant - read_half(block + 2) * minimum;
}

__device__ float warp_sum(float value) {
    for (int shift = 16; shift > 0; shift /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, shift);
    }
    return value;
}

template<bool max_value>
__device__ float block_reduce(float value, float* scratch) {
    const int lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    for (int stride = block_threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            scratch[lane] = max_value
                ? fmaxf(scratch[lane], scratch[lane + stride])
                : scratch[lane] + scratch[lane + stride];
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void embedding(const unsigned char* weights, float* output,
    int columns, int row_bytes, int type, int token) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < columns) {
        output[column] = unpack(weights + (unsigned long long)token * row_bytes, column, type);
    }
}

extern "C" __global__ void matvec(const unsigned char* weights, const float* input, float* output,
    int columns, int rows, int row_bytes, int type) {
    const int row = blockIdx.x * warps_per_block + threadIdx.x / warp_width;
    if (row >= rows) return;
    const int lane = threadIdx.x % warp_width;
    const unsigned char* packed = weights + (unsigned long long)row * row_bytes;
    float sum = 0.0f;
    for (int column = lane; column < columns; column += warp_width) {
        sum += unpack(packed, column, type) * input[column];
    }
    sum = warp_sum(sum);
    if (lane == 0) output[row] = sum;
}

extern "C" __global__ void rms_norm(const float* input, const float* weights, float* output,
    int length, float epsilon) {
    __shared__ float scratch[block_threads];
    float sum = 0.0f;
    for (int index = threadIdx.x; index < length; index += block_threads) {
        sum += input[index] * input[index];
    }
    const float scale = rsqrtf(block_reduce<false>(sum, scratch) / length + epsilon);
    for (int index = threadIdx.x; index < length; index += block_threads) {
        output[index] = input[index] * scale * weights[index];
    }
}

extern "C" __global__ void prepare_rope(const float* frequencies, float* rotation,
    int half, int position, float scale) {
    const int index = threadIdx.x;
    if (index < half) {
        float sine, cosine;
        sincosf(position * frequencies[index], &sine, &cosine);
        rotation[index] = cosine * scale;
        rotation[half + index] = sine * scale;
    }
}

extern "C" __global__ void rope_and_cache(float* qkv, float* keys, float* values, const float* rotation,
    int embedding_size, int kv_size, int head_size, int half, int position) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int element = index % head_size;
    if (index < embedding_size && element < half) {
        const float first = qkv[index];
        const float second = qkv[index + half];
        const float cosine = rotation[element];
        const float sine = rotation[half + element];
        qkv[index] = first * cosine - second * sine;
        qkv[index + half] = second * cosine + first * sine;
    }
    if (index < kv_size) {
        const float* key = qkv + embedding_size;
        float value = key[index];
        if (element < half) {
            value = value * rotation[element] - key[index + half] * rotation[half + element];
        } else if (element < 2 * half) {
            value = value * rotation[element - half] + key[index - half] * rotation[element];
        }
        keys[position * kv_size + index] = value;
        values[position * kv_size + index] = qkv[embedding_size + kv_size + index];
    }
}

extern "C" __global__ void attention(const float* queries, const float* keys, const float* values,
    float* result, int head_size, int kv_size, int heads_per_group, int length, float scale) {
    extern __shared__ float scores[];
    __shared__ float scratch[block_threads];
    const int head = blockIdx.x;
    const int thread = threadIdx.x;
    const int lane = thread % warp_width;
    const int kv_offset = (head / heads_per_group) * head_size;
    const float* query = queries + head * head_size;
    for (int past = thread / warp_width; past < length; past += warps_per_block) {
        const float* key = keys + past * kv_size + kv_offset;
        float score = 0.0f;
        for (int i = lane; i < head_size; i += warp_width) score += query[i] * key[i];
        score = warp_sum(score);
        if (lane == 0) scores[past] = score * scale;
    }
    __syncthreads();

    float maximum = -3.402823466e38F;
    for (int past = thread; past < length; past += block_threads) maximum = fmaxf(maximum, scores[past]);
    maximum = block_reduce<true>(maximum, scratch);
    float sum = 0.0f;
    for (int past = thread; past < length; past += block_threads) {
        const float probability = expf(scores[past] - maximum);
        scores[past] = probability;
        sum += probability;
    }
    const float inverse = 1.0f / block_reduce<false>(sum, scratch);
    for (int i = thread; i < head_size; i += block_threads) {
        float total = 0.0f;
        for (int past = 0; past < length; past++) {
            total += scores[past] * inverse * values[past * kv_size + kv_offset + i];
        }
        result[head * head_size + i] = total;
    }
}

extern "C" __global__ void residual(float* hidden, const float* projected, int length) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) hidden[i] += projected[i];
}

extern "C" __global__ void swiglu(const float* gate_up, float* output, int length) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        const float gate = gate_up[i];
        output[i] = (gate / (1.0f + expf(-gate))) * gate_up[length + i];
    }
}
