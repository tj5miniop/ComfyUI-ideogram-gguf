# CLIPLoader GGUF

Loads one text encoder from a GGUF file and returns a ComfyUI `CLIP`.

## Supported Models

This node handles supported GGUF text model architectures such as T5, LLaMA-style text encoders, Qwen variants, and Gemma 3 variants when their tensor types are supported by this loader.

## Unsupported K Quants

`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` are rejected with an explanatory error.

Use non-K GGUF quant types such as `Q4_0`, `Q5_0`, `Q5_1`, or `Q8_0`, or use a dedicated accelerated backend for K-quant models.

## Usage

Place supported `.gguf` text encoders in a ComfyUI text encoder or CLIP model folder, then select the file and matching CLIP type.
