# Unet Loader GGUF

Loads a diffusion model from a GGUF file and returns a ComfyUI `MODEL`.

## Supported GGUF Tensor Types

This loader supports non-K GGUF tensor types that can be handled efficiently by the PyTorch fallback path:

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

## Unsupported K Quants

`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` are intentionally not supported.

Those formats are useful in runtimes with fused quantized matrix multiplication kernels. This node uses PyTorch linear operations, so `_K` weights must be dequantized before every linear operation. That made inference very slow and memory intensive.

If a `_K` GGUF is selected, the loader stops early and reports the unsupported tensor types and example tensor names.

## Usage

Place supported `.gguf` diffusion models in a ComfyUI diffusion model or UNet model folder, then select the file in this node.

Use a non-K quant such as `Q4_0`, `Q5_0`, `Q5_1`, or `Q8_0` when using this loader.

## BF16 and FP16

On GPUs where BF16 is not fast for this workload, the loader asks ComfyUI to run the diffusion model in FP16. This avoids the slow BF16-to-FP32 manual cast path that can appear on consumer Ampere GPUs.
