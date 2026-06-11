# Method - Nova HackAIthon 2026 Bang C CPU Submission

## Objective and architecture

The final submission is an offline, self-contained CPU Docker image for Vietnamese multiple-choice questions:

```text
viape/hackaithon-c:v9-3-final
```

```text
/data/private_test.csv or /data/public_test.csv
  -> schema normalization and qid preservation
  -> lightweight question-risk assessment
  -> grammar-constrained Qwen3.5 GGUF call
  -> optional single verifier call for high-risk rows
  -> strict A/B/C/D validation
  -> /output/pred.csv
```

Low-risk rows receive exactly one model call. High-risk rows receive one direct call plus at most one verifier call. The official CPU path has no permutation, pairwise comparison, multi-call voting loops, or heuristic fallback.

## Model and backend

- Qwen3.5-9B Q4_K_M GGUF, within the contest model-size limit.
- `llama-cpp-python==0.3.28` CPU backend.
- GPU layers forced to zero with `N_GPU_LAYERS=0`.
- GGUF chat template and llama.cpp grammar constrain output to `A/B/C/D`.
- The model is embedded in the published Docker image.

A 4B model was allowed only if it was at least 1.7x faster while losing no more than 3.0 accuracy points on an approved labeled development set. VMLU was excluded because the separately downloaded dataset license is still marked to be updated. With no approved benchmark proving the 4B threshold, the conservative selection rule kept 9B.

The public test was used only for runtime and output-format validation. Its answers were not asserted, trained on, or hard-coded.

## RAG decision

RAG is disabled because no approved labeled benchmark demonstrated at least 1.0 accuracy-point improvement within a 10 percent latency increase. This also removes unused dependencies from the final runtime.

## Strict CPU configuration

```text
CPU_PORTABLE=1
FORCE_CPU=1
N_GPU_LAYERS=0
ENABLE_NETWORK=0
STRICT_NO_FALLBACK=1
REQUIRE_MODEL=1
ALLOW_HEURISTIC=0
ENABLE_RAG=0
USE_KNOWLEDGE_ENGINE=0
ENABLE_VENDOR_RAG_FUSION=0
```

The minimal image contains `llama_cpp` and its required NumPy dependency. It does not contain Torch, Transformers, bm25s, CUDA, or GPU runtime libraries.

## Validation

The source passed compileall, CLI help, strict local GGUF execution, the full 90-test suite, and Docker execution with `--network none`.

The pushed tag was pulled back from Docker Hub and executed three independent times with `--network none`. Every run exited successfully, produced exactly `qid,answer`, preserved qid order, and produced only `A/B/C/D` answers.

```text
Platform: linux/amd64
Digest: sha256:ffcbeeb9a600f6f4820dc01868b2af6ee7d2024fd1f4a929a0a155daec8c380a
```