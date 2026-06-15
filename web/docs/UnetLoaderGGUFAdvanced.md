# Unet Loader GGUF Advanced

Loads a diffusion GGUF model with explicit control over dequantization and patch dtypes.

## Parameters

- **unet_name**: GGUF diffusion model file.
- **dequant_dtype**: Dtype used when a quantized tensor is expanded for PyTorch operations.
- **patch_dtype**: Dtype used when LoRA or other model patches are applied.
- **patch_on_device**: Moves patches to the model load device before applying them.

## Supported GGUF Tensor Types

Supported tensor types are `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`, and `IQ4_XS`.

`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` are rejected before loading. These formats need fused quantized linear kernels to be fast; this node uses PyTorch linear operations after dequantization.

## Notes

Use the default dtype settings unless you are benchmarking or debugging a specific model.

On GPUs where BF16 is not fast for this workload, the loader asks ComfyUI to run the diffusion model in FP16 and loads BF16 storage tensors as FP16 where possible. Small one-dimensional norm and scale tensors may remain FP32 for compatibility.
