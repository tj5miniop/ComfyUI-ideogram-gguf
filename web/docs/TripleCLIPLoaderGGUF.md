# TripleCLIPLoader GGUF

Loads three text encoders and returns one ComfyUI `CLIP` object.

## Usage

Use this node for workflows that require three text encoder files, such as SD3-style setups.

Each GGUF file is checked before loading. Unsupported `_K` quant files are rejected early with tensor type counts and example tensor names.

## Supported Quantization

Supported quantized GGUF tensor types include `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`, and `IQ4_XS`.

`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` are not supported.
