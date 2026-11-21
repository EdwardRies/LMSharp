# Model compatibility and verification

## Verified model file

LMSharp targets the [EasierAI Q4_K_L conversion](https://huggingface.co/EasierAI/Phi-4-Mini-3.8B/blob/main/Phi-4-Mini-3.8B-Q4_K_L.gguf) of [Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct).

| Property | Verified value |
| --- | --- |
| File | `Phi-4-Mini-3.8B-Q4_K_L.gguf` |
| File size | 2,640,722,080 bytes |
| SHA-256 | `290E7A9D82926FF260EC537CD24F7223D630D9C5A3027A7D8068050B035D76C2` |
| GGUF format | Version 3, little endian |
| Architecture identifier | `phi3` |
| Tensor inventory | 67 F32, 80 Q4_K, 32 Q5_K, 16 Q6_K, 1 Q8_0 |
| Transformer | 32 layers, 24 query heads, 8 key/value heads |
| Embedded tokenizer | 200,064 entries, 199,742 merge rules |
| LMSharp context limit | 4,096 input and output tokens combined |

The GGUF reports a maximum context of 131,072 positions. LMSharp implements the short-context LongRoPE profile, limited to 4,096 positions. The loader checks the architecture, tensor shapes and storage types, and expected tokenizer profile. Other GGUF files may contain a different network, tokenizer, chat template, or quantization. The filename alone cannot establish compatibility.

The runtime reads the tokenizer directly from this GGUF. Its base vocabulary IDs and merge ranks matched the Microsoft tokenizer files used for comparison. The larger embedded vocabulary also includes 10 added chat/tool tokens that Microsoft lists separately and 35 reserved padding entries.

## Running the integration tests

The default .NET test command needs no model assets. To run the tests against the real model, set `LMSHARP_PHI4_GGUF` to its full path. Set `LMSHARP_TEST_CUDA=1` as well to enable the GPU tests on a machine with [CUDA configured](Usage.md#cuda-setup).

```powershell
$env:LMSHARP_PHI4_GGUF = (Resolve-Path './resources/models/Phi-4-Mini-3.8B-Q4_K_L.gguf').Path
dotnet test LMSharp.slnx -c Release

# Include CUDA tests on a configured machine.
$env:LMSHARP_TEST_CUDA = '1'
dotnet test LMSharp.slnx -c Release
```

The test suite reports an explicit skip for each integration test whose opt-in variables are absent. The asserted expectations are in C# and require no Python packages. The [optional numerical oracle](../tests/reference/phi4_float_reference.py) uses NumPy and the Python GGUF reader to regenerate reference values during development.

## What the tests verify

| Test | Evidence |
| --- | --- |
| [Tensor access](../tests/LMSharp.UnitTests/Phi4TensorIntegrationTests.cs) | All 196 payload ranges validate. Selected offsets, lengths, and raw bytes match Python `gguf` 0.19.0, including a read above 2 GiB. |
| [Tokenizer](../tests/LMSharp.UnitTests/Phi4TokenizerTests.cs) | Token IDs, decoded text, and chat prompts match fixtures from Hugging Face `tokenizers` 0.23.3 and Jinja2 3.1.6. |
| [CPU kernels](../tests/LMSharp.UnitTests/CpuKernelsTests.cs) | One real row per storage type matches independent dequantization, and matrix-vector products match reference values. |
| [Forward pass](../tests/LMSharp.UnitTests/Phi4InferenceTests.cs) | Selected first-layer values and final logits match the dequantized float32 reference. Cached continuation matches replay. |
| [Generation](../tests/LMSharp.UnitTests/Phi4GenerationTests.cs) | Two greedy fixtures match [llama.cpp b12608](https://github.com/ggml-org/llama.cpp/releases/tag/b12608), including a reply that stops on an end marker. |
| [Chat](../tests/LMSharp.UnitTests/Phi4ChatTests.cs) | A follow-up recalls an earlier codeword and settings updates preserve or reset the conversation as advertised. |

## Verified generation

For `What is 2 + 2?` with a 12-token budget, the response is **The answer to 2 + 2 is 4.** The output IDs are `976 6052 316 220 17 659 220 17 382 220 19 13`. These match the corresponding reference run. With a 32-token budget, LMSharp and llama.cpp matched through the first 17 tokens and then chose different wording. Different numerical paths can produce different choices when scores are close.

The float32 KV cache uses 256 KiB per position across all 32 layers. A 1,024-token context allocates 256 MiB for the cache. A 4,096-token context allocates 1 GiB, in addition to model weights and working buffers.

## Verified CUDA execution

[CUDA integration tests](../tests/LMSharp.UnitTests/Phi4CudaTests.cs) ran with Windows x64, an RTX 5080, NVIDIA driver 616.92, and NVRTC 13.4.92. The maximum observed absolute differences between CPU and CUDA logits were about 0.000031 after prefill and 0.000046 after one cached token. CUDA matched the 12-token greedy response, recalled the codeword in chat, and switched CPU → CUDA → CPU within one terminal session.

On that system, idle chat used about 3,000 MiB of additional VRAM with 1,024 positions or 3,768 MiB with 4,096 positions. The driver reported a return to baseline after exit. These figures are device-wide snapshots, not a minimum hardware specification.

In warmed Release runs for the 11-input-token, 12-output-token arithmetic example, CPU decoding measured 6.23 tokens/s on a Ryzen 9 9950X3D. Two CUDA runs on the RTX 5080 measured 94.31–94.65 tokens/s after the first token. These measurements describe that workload and hardware.

See [third-party notices](../THIRD-PARTY-NOTICES.md) for model and reference sources.
