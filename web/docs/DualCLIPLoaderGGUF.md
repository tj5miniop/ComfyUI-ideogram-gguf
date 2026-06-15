# DualCLIPLoader GGUF

Loads two text encoders and returns one ComfyUI `CLIP` object.

## Usage

Select two supported text encoder files and the matching CLIP type for the target workflow.

Each GGUF file is validated independently. If either file contains unsupported `_K` tensors, loading stops and the error reports the unsupported tensor types and example tensor names.

## Supported Quantization

Use non-K GGUF quant types such as `Q4_0`, `Q5_0`, `Q5_1`, or `Q8_0`.

`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` are not supported in this PyTorch-based loader.
