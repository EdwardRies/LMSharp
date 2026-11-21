# LMSharp

LMSharp is a local large language model inference engine for learning how an LLM turns a prompt into a response. It is written for developers who know C# and want to follow the entire process in a familiar language, from reading model weights to choosing the next token. You can build, run, and follow the core implementation with .NET 10 and no Python setup.

The console runs the verified Phi-4 Mini GGUF on the CPU or, on Windows x64, on an NVIDIA GPU through optional CUDA kernels. LMSharp implements the inference loop, including tokenization, the transformer forward pass, and sampling. The app runs locally without a model server or an API key.

## Run it

1. Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).
2. Download [Phi-4-Mini-3.8B-Q4_K_L.gguf](https://huggingface.co/EasierAI/Phi-4-Mini-3.8B/blob/main/Phi-4-Mini-3.8B-Q4_K_L.gguf) and place it in `resources/models/`. The file is about 2.64 GB and includes the weights and tokenizer data. See [model compatibility](docs/Model-Compatibility.md) for its SHA-256.
3. Open a terminal at the repository root and start chat.

```text
dotnet run --project src/LMSharp.Console -c Release --no-launch-profile -- chat --model resources/models/Phi-4-Mini-3.8B-Q4_K_L.gguf
```

Enter a question at `You>`. Chat keeps context across turns and displays how much of the window is in use. Type `/help` for commands, `/settings` to see active options, or `/exit` to quit.

For a repeatable one-shot example, run the following command.

```text
dotnet run --project src/LMSharp.Console -c Release --no-launch-profile -- generate --model resources/models/Phi-4-Mini-3.8B-Q4_K_L.gguf --prompt "What is 2 + 2?" --max-tokens 12
```

The verified greedy response is **The answer to 2 + 2 is 4.** The console also reports the generated token IDs and timing so you can trace the result.

## Learn from the code

- [How inference works](docs/Architecture.md) traces a prompt through the tokenizer, transformer, attention cache, and sampler. Start here to follow execution through the C# source.
- [Terminal usage](docs/Usage.md) covers chat controls, inspection, Visual Studio, and switching between CPU RAM and CUDA VRAM.
- [Model compatibility and verification](docs/Model-Compatibility.md) identifies the supported GGUF, describes the test fixtures, and records CPU and CUDA results.

LMSharp focuses on one model profile and text-only inference. The context defaults to 1,024 tokens and supports up to 4,096. CPU mode is the default. See the [CUDA setup guide](docs/Usage.md#cuda-setup) for GPU requirements.

## Build and test

```text
dotnet build LMSharp.slnx -c Release
dotnet test LMSharp.slnx -c Release --no-build
```

The default suite runs without the model or a GPU. The [verification guide](docs/Model-Compatibility.md#running-the-integration-tests) explains how to enable the real-model and CUDA tests. The optional NumPy reference script is a development tool separate from the application.

## License

A license for the LMSharp source has not been specified. The model and other upstream materials have separate terms. See the [third-party notices](THIRD-PARTY-NOTICES.md).
