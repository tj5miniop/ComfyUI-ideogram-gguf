# QuadrupleCLIPLoader GGUF

Loads four text encoders and returns one ComfyUI `CLIP` object.

## Usage

Use this node only for workflows that explicitly require four text encoder files.

Each GGUF file is validated before loading. If a selected file contains unsupported `_K` tensors, the loader reports the affected quant types and example tensor names.

## Supported Quantization

Use non-K GGUF quant types such as `Q4_0`, `Q5_0`, `Q5_1`, or `Q8_0`.

`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` are not supported because this loader does not provide fused K-quant linear kernels.
