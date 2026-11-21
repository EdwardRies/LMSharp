# Terminal usage

Start at the repository root with the [supported GGUF](Model-Compatibility.md#verified-model-file) in `resources/models/`. The examples use a Release build for normal use. Pass a different path with `--model` if you store your copy elsewhere.

## Interactive chat

```text
dotnet run --project src/LMSharp.Console -c Release --no-launch-profile -- chat --model resources/models/Phi-4-Mini-3.8B-Q4_K_L.gguf
```

Type at `You>` and press Enter. You can follow up, and the model can attend to earlier turns in that session. Try asking it to remember the codeword `ORCHID`, then ask for the codeword without repeating it. Use `/help` to list commands and `/settings` to see the active values.

| What to change | Launch option | In-chat command |
| --- | --- | --- |
| Model file | `--model <path>` | `/model <path>` |
| Compute device | `--device cpu` or `--device cuda` | `/device cpu`, `/device cuda`, or `/device toggle` |
| Context window | `--context 4096` | `/context 4096` |
| Instructions for the model | `--system "Keep replies brief."` | `/system Keep replies brief.` or `/system` to clear |
| CPU workers | `--threads 8` | `/threads 8` or `/threads default` |
| Per-reply token budget | `--max-tokens 128` | `/max-tokens 128` |
| Sampling temperature | `--temperature 0.7` | `/temperature 0.7` |
| Sampling seed | `--seed 123` | `/seed 123` or `/seed random` |

Changes to reply length, temperature, and seed preserve the conversation. Changing the model, device, context size, or system prompt starts a fresh conversation. Changing CPU worker count also resets context in CPU mode. In CUDA mode it sets the worker count for the next switch to CPU. When a replacement model or device fails to load, the active conversation stays available.

Chat defaults to a **1,024-token context** and a **64-token reply budget**. The context can be increased to **4,096** and includes both sides of the conversation. A reply can stop at its token budget, at the remaining context capacity, or when the model emits an end marker. Use `/reset` to clear context or `/exit` to leave. Live changes apply to the current run. Launch options determine the next run's settings.

## Visual Studio

Open `LMSharp.slnx`, set `LMSharp.Console` as the startup project, and choose **Release**. Set the project's launch arguments to `chat --model "<absolute path to the GGUF>"`. Press **Ctrl+F5** to start without debugging. The startup line identifies the selected device.

## CUDA setup

CPU mode uses system RAM and starts with the .NET SDK and model alone. CUDA mode uses VRAM and requires **Windows x64**, a compatible NVIDIA driver, and [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html). A CUDA Toolkit installation can supply NVRTC through `CUDA_PATH`. If using an explicit installation, set `LMSHARP_NVRTC_DLL` to the compiler DLL and keep its matching NVRTC built-ins alongside it.

NVIDIA also provides an optional [per-user wheel installation](https://docs.nvidia.com/cuda/cuda-quick-start-guide/#pip-wheels-windows). The following installer command was verified with driver 616.92 on the RTX 5080. It uses Python to install the compiler DLLs. LMSharp itself loads those DLLs from C#.

```powershell
python -m pip install --target (Join-Path $env:LOCALAPPDATA 'LMSharp\cuda-runtime') 'nvidia-cuda-nvrtc==13.4.92'
```

Launch with `--device cuda` or type `/device cuda` in an open chat. Use `/device cpu` to return to system RAM. See [verified CUDA execution](Model-Compatibility.md#verified-cuda-execution) for observed device memory use.

## Inspect, tokenize, or generate once

The following commands expose separate parts of the inference pipeline.

```text
dotnet run --project src/LMSharp.Console -c Release --no-launch-profile -- inspect --model resources/models/Phi-4-Mini-3.8B-Q4_K_L.gguf
dotnet run --project src/LMSharp.Console -c Release --no-launch-profile -- inspect --model resources/models/Phi-4-Mini-3.8B-Q4_K_L.gguf --tensor blk.31.ffn_up.weight
dotnet run --project src/LMSharp.Console -c Release --no-launch-profile -- tokenize --model resources/models/Phi-4-Mini-3.8B-Q4_K_L.gguf --text "Hello, world!"
dotnet run --project src/LMSharp.Console -c Release --no-launch-profile -- generate --model resources/models/Phi-4-Mini-3.8B-Q4_K_L.gguf --prompt "What is 2 + 2?" --max-tokens 12
```

- `inspect` prints the model architecture, tensor count, and metadata. With `--tensor`, it prints the named tensor's dimensions, GGML type ID, file offset, length, and first bytes.
- `tokenize` prints token IDs for plain text. The example produces `13225 11 2375 0`.
- `generate` formats a single prompt and prints its reply. Generated IDs, timings, and the stop reason go to standard error. It accepts the same sampling and device options as chat and uses a context sized for the prompt and requested output.
