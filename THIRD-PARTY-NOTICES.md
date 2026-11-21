# Third-party notices

- **Microsoft Phi-4-mini-instruct.** LMSharp implements the text chat format and pre-tokenization rules published with the model. The pre-tokenization pattern is from the upstream tokenizer configuration. See the [model card](https://huggingface.co/microsoft/Phi-4-mini-instruct) and the [Microsoft MIT license](https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/LICENSE).
- **EasierAI GGUF conversion.** The [verified GGUF](https://huggingface.co/EasierAI/Phi-4-Mini-3.8B/blob/main/Phi-4-Mini-3.8B-Q4_K_L.gguf) is a separate download. Its publisher lists it under the MIT license. The model weights and standalone tokenizer files are distributed by their respective publishers.
- **GGML and llama.cpp.** The [GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [llama.cpp](https://github.com/ggml-org/llama.cpp) are references for file format, quantized layouts, and independent inference checks. See the [upstream MIT license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE).
- **NVIDIA NVRTC.** CUDA mode uses NVIDIA's separately installed runtime compiler. See the [NVIDIA documentation](https://docs.nvidia.com/cuda/nvrtc/index.html) for the runtime and its terms.

## Upstream MIT notices

- Phi-4 tokenizer pattern. Copyright (c) Microsoft Corporation.
- GGML dequantization references. Copyright (c) 2023-2026 The ggml authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
