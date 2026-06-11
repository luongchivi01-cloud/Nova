# Model Notice

## Exact submitted model

The final Docker image `viape/hackaithon-c:v9-3-final` contains:

```text
Qwen3.5-9B-Q4_K_M.gguf
SHA-256: 03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8
Size: 5680522464 bytes
```

The checksum and size match the following published file:

https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf

The corresponding Hugging Face Git-LFS pointer publishes the same SHA-256 and
size:

https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/raw/main/Qwen3.5-9B-Q4_K_M.gguf

## Use in this project

- The model is used locally through `llama-cpp-python`.
- The model is not fine-tuned or modified by this project.
- The model is not committed to GitHub.
- Inference runs offline and does not call an external model API.
- The final profile forces `N_GPU_LAYERS=0`.

## License

Qwen open-weight models are distributed under the Apache License 2.0. A copy
is stored at `licenses/APACHE-2.0-QWEN.txt`. Users remain responsible for
following the model license and acceptable-use requirements published by the
model distributor and original model owner.
