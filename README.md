# ComfyUI-GGUF

GGUF loader nodes for native ComfyUI diffusion models and text encoders.

This fork focuses on loading non-K GGUF quantized transformer/DiT models through ComfyUI's normal model patching path, with added support for Ideogram-style GGUF diffusion models.

GGUF is the model file format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp). It is useful for transformer-heavy diffusion models where quantized linear weights can reduce storage and VRAM use.

## Installation

Install this fork into your ComfyUI custom nodes directory:

```bash
git clone https://github.com/molbal/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
```

Then install the requirements in the same Python environment that runs ComfyUI:

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
```

For ComfyUI portable on Windows, run this from the portable root folder:

```bat
git clone https://github.com/molbal/ComfyUI-GGUF ComfyUI\custom_nodes\ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

Restart ComfyUI after installation or update.

## Nodes

The loader nodes are in the `bootleg` category.

Diffusion model loaders:

- `Unet Loader (GGUF)`
- `Unet Loader (GGUF/Advanced)`

Text encoder loaders:

- `CLIPLoader (GGUF)`
- `DualCLIPLoader (GGUF)`
- `TripleCLIPLoader (GGUF)`
- `QuadrupleCLIPLoader (GGUF)`

The advanced UNet loader exposes explicit dtype controls for dequantization and patch application. Use the defaults unless you are benchmarking or debugging.

## Model Folders

Diffusion `.gguf` files are discovered from ComfyUI diffusion model folders:

```text
ComfyUI/models/diffusion_models/
ComfyUI/models/unet/
```

Text encoder `.gguf` files are discovered from:

```text
ComfyUI/models/text_encoders/
ComfyUI/models/clip/
```

## Supported GGUF Tensor Types

This fork supports GGUF tensor types that can be handled by the PyTorch dequantization path:

- `F32`
- `F16`
- `BF16`
- `Q4_0`
- `Q4_1`
- `Q5_0`
- `Q5_1`
- `Q8_0`
- `IQ4_NL`
- `IQ4_XS`

K-quants are intentionally rejected before loading:

- `Q2_K`
- `Q3_K`
- `Q4_K`
- `Q5_K`
- `Q6_K`

Those formats need fused quantized linear kernels to be fast. This node pack dequantizes for PyTorch operations, so K-quants are slow and memory-heavy in this path. Use non-K files such as `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, or `Q8_0`.

## Ideogram 4 GGUF

Ideogram 4 support in this fork is intended for the non-K GGUF files published here:

- [molbal/ideogram-4-gguf](https://huggingface.co/molbal/ideogram-4-gguf)

Ideogram 4 uses two diffusion model components:

| Component | File pattern | Purpose |
| --- | --- | --- |
| Main transformer | `ideogram4-transformer-*.gguf` | Text-guided diffusion model |
| Unconditional transformer | `ideogram4-unconditional_transformer-*.gguf` | Unconditional/CFG counterpart |

Place both files in `models/diffusion_models` or `models/unet`, then load them with `Unet Loader (GGUF)` or `Unet Loader (GGUF/Advanced)` in an Ideogram 4 workflow that accepts separate main and unconditional diffusion models.

The main and unconditional model do not have to use the same quant level. Common pairings:

| Main transformer | Unconditional transformer | Notes |
| --- | --- | --- |
| `q8_0` | `q8_0` | Highest quality baseline |
| `q8_0` | `q4_0` | Saves memory mostly on the CFG side |
| `q5_1` | `q4_1` | Balanced quality and size |
| `q5_0` | `q4_0` | Lower memory starting point |
| `q4_0` | `q4_0` | Smallest available pair |

The GGUF files are only the diffusion transformers. Your workflow still needs the other Ideogram 4 runtime assets expected by ComfyUI, such as the text encoder or multimodal encoder and VAE.

## Other Pre-Quantized Models

Upstream ComfyUI-GGUF model repositories may still be useful when the files use tensor types supported by this fork:

- [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf)
- [city96/stable-diffusion-3.5-large-gguf](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)
- [city96/stable-diffusion-3.5-large-turbo-gguf](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)
- [city96/t5-v1_1-xxl-encoder-gguf](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)

If a model fails immediately with an unsupported K-quant error, use a non-K variant instead.

## LoRA Notes

LoRA loading is experimental with quantized GGUF weights, but the built-in ComfyUI LoRA loader nodes should work for supported model families. The advanced loader can move patches to the model load device before applying them when needed.

## Conversion

The `tools` directory contains helper scripts for creating GGUF files from model weights. For Ideogram 4, the published files were converted from the original FP8 checkpoint by expanding FP8 scaled weights to BF16 and then converting/quantizing with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

For best compatibility with this fork, produce non-K quant types: `q4_0`, `q4_1`, `q5_0`, `q5_1`, or `q8_0`.

## Credits

This fork builds on the original [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) project and the GGUF/GGML ecosystem.
