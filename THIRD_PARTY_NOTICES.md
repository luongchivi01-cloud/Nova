# Third-Party Notices

This project uses open-source software and an open-weight model. The final
Docker image does not vendor the optional research repositories mentioned by
some compatibility modules in `src/`.

## Qwen3.5-9B model

- Original model family: `Qwen/Qwen3.5-9B`
- GGUF distributor: `unsloth/Qwen3.5-9B-GGUF`
- Exact file: `Qwen3.5-9B-Q4_K_M.gguf`
- Source URL:
  https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
- Local SHA-256:
  `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`
- Size: `5680522464` bytes
- License: Apache License 2.0

The model is used without fine-tuning. It is included only in the published
self-contained Docker image and is not committed to this GitHub repository.
See `licenses/APACHE-2.0-QWEN.txt`.

## llama-cpp-python

- Project: https://github.com/abetlen/llama-cpp-python
- Version in final image: `0.3.28`
- License: MIT
- Copyright: Andrei Betlen and contributors

See `licenses/MIT-LLAMA-CPP-PYTHON.txt`.

## llama.cpp

`llama-cpp-python` includes/binds the llama.cpp runtime.

- Project: https://github.com/ggml-org/llama.cpp
- License: MIT
- Copyright: Georgi Gerganov and contributors

See `licenses/MIT-LLAMA-CPP.txt`.

## Optional compatibility names

The source repository contains optional adapters or compatibility detection for
projects such as bm25s, VnCoreNLP, Transformers, vLLM, FlashRAG, txtai,
GraphRAG, LightRAG, DSPy, and Outlines. These projects are not installed,
vendored, or activated in the final CPU image. Their names are retained only
to describe optional interfaces and prior experiments.

## Contest dataset

The public-test JSON and generated prediction CSV are not committed to this
repository. They are used only for the HackAIthon submission workflow under
the competition organizer's terms.
