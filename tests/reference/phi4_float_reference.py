"""Optional float32 oracle for LMSharp's short-context Phi-4 Mini profile.

Requires NumPy and the gguf Python package. Run with the model path followed by
the exact prompt token IDs. The runtime and C# test suite do not invoke this tool.
"""

import argparse
import json

import numpy as np
from gguf import GGUFReader, dequantize


def reference(model_path, tokens):
    reader = GGUFReader(model_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    def load(name, indices=slice(None)):
        tensor = tensors[name]
        return dequantize(tensor.data[indices], tensor.tensor_type)

    def rms_norm(x, weights):
        mean_square = np.square(x.astype(np.float64)).mean(axis=-1, keepdims=True)
        return x / np.sqrt(mean_square + 1e-5).astype(np.float32) * weights

    hidden = load("token_embd.weight", tokens)
    positions = np.arange(len(tokens), dtype=np.float32)
    frequencies = np.power(np.float32(10000), -np.arange(48, dtype=np.float32) / 48)
    angles = positions[:, None] * frequencies[None, :] / load("rope_factors_short.weight")
    cosines = np.cos(angles)[:, None, :] * np.float32(1.1902381)
    sines = np.sin(angles)[:, None, :] * np.float32(1.1902381)
    causal_mask = np.triu(np.ones((len(tokens), len(tokens)), dtype=bool), k=1)

    def rotate(heads):
        first = heads[:, :, :48].copy()
        second = heads[:, :, 48:96].copy()
        heads[:, :, :48] = first * cosines - second * sines
        heads[:, :, 48:96] = second * cosines + first * sines
        return heads

    layer_samples = []
    for layer in range(32):
        prefix = f"blk.{layer}."
        attention_input = rms_norm(hidden, load(prefix + "attn_norm.weight"))
        qkv = attention_input @ load(prefix + "attn_qkv.weight").T
        q, k, v = np.split(qkv, [3072, 4096], axis=-1)
        query = rotate(q.reshape(-1, 24, 128))
        key = rotate(k.reshape(-1, 8, 128)).repeat(3, axis=1)
        value = v.reshape(-1, 8, 128).repeat(3, axis=1)
        scores = np.einsum("tqd,sqd->qts", query, key) / np.sqrt(np.float32(128))
        scores[:, causal_mask] = -np.inf
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        attention = np.einsum("qts,sqd->tqd", probabilities, value).reshape(-1, 3072)
        hidden += attention @ load(prefix + "attn_output.weight").T

        mlp_input = rms_norm(hidden, load(prefix + "ffn_norm.weight"))
        gate, up = np.split(mlp_input @ load(prefix + "ffn_up.weight").T, 2, axis=-1)
        activated = gate / (1 + np.exp(-gate)) * up
        hidden += activated @ load(prefix + "ffn_down.weight").T
        if layer == 0:
            indices = [0, 1, 2, 47, 48, 95, 96, 127, 128, 513, 1234, 2047, 3071]
            layer_samples = hidden[:2, indices].tolist()

    final = rms_norm(hidden[-1], load("output_norm.weight"))
    vocabulary_size = int(tensors["token_embd.weight"].shape[1])
    logits = np.empty(vocabulary_size, dtype=np.float32)
    for start in range(0, vocabulary_size, 4096):
        end = min(start + 4096, vocabulary_size)
        logits[start:end] = load("token_embd.weight", slice(start, end)) @ final

    top_ids = np.argsort(logits)[-10:][::-1].tolist()
    return {
        "layer_0": layer_samples,
        "top_logits": [[index, float(logits[index])] for index in top_ids],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("tokens", type=int, nargs="+")
    args = parser.parse_args()
    print(json.dumps(reference(args.model, args.tokens), indent=2))
