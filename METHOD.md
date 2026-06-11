# Method - Nova HackAIthon 2026 Bang C CPU Submission

## 1. Objective

The final submission is an offline, self-contained CPU Docker image for
Vietnamese multiple-choice questions. It prioritizes a reproducible judge
contract, bounded model calls, and strict output validity.

Final image:

```text
viape/hackaithon-c:v9-3-final
```

## 2. Runtime architecture

```text
/data/private_test.csv or /data/public_test.csv
  -> schema normalization and qid preservation
  -> lightweight question-risk assessment
  -> grammar-constrained Qwen3.5 GGUF call
  -> optional single verifier call for high-risk rows
  -> strict A/B/C/D validation
  -> /output/pred.csv
```

The official CPU path is deliberately narrow:

- Low-risk row: exactly one model call.
- High-risk row: one direct call plus at most one verifier call.
- No permutation, pairwise comparison, or multi-call voting loops.
- No heuristic fallback in official mode.

## 3. Model and backend

- Model family: Qwen3.5, within the contest limit of 9B parameters.
- Quantization: Q4_K_M GGUF.
- Backend: `llama-cpp-python==0.3.28` CPU wheel.
- GPU layers: forced to zero with `N_GPU_LAYERS=0`.
- Output: a llama.cpp grammar restricts generation to one label from
  `A/B/C/D`.
- The model's GGUF chat template is used through the chat-completion API.

The model is embedded in the Docker image. The judge does not need a model
mount or a network connection.

The exact model source, size, and SHA-256 checksum are recorded in
`MODEL_NOTICE.md`. Third-party licenses and attribution are recorded in
`THIRD_PARTY_NOTICES.md` and `licenses/`.

## 4. Model selection decision

The agreed selection rule allowed a 4B model only if it was at least 1.7 times
faster than 9B while losing no more than 3.0 accuracy percentage points on an
approved labeled development set.

VMLU was not used because its repository license covers repository contents,
while the separately downloaded dataset is still marked with licensing to be
updated. No other approved labeled benchmark was available to prove that the
4B model met the accuracy threshold. The conservative selection rule therefore
kept Qwen3.5-9B Q4_K_M.

The public test was used only as unlabeled inference input and for runtime and
output-format validation. Its answers were not supplied, asserted, trained on,
or hard-coded.

## 5. RAG decision

RAG is disabled in the final image. No approved labeled benchmark demonstrated
the required accuracy improvement of at least 1.0 percentage point within the
allowed latency increase of at most 10 percent. Disabling RAG also removes
unused runtime dependencies and makes the final execution path easier to
reproduce.

## 6. Strictness and portability

The final image sets:

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

The minimal image contains `llama_cpp` and its required NumPy dependency. It
does not contain Torch, Transformers, bm25s, CUDA, or GPU runtime libraries.

## 7. Validation

Before publishing, the source passed:

- `python -m compileall -q src`
- CLI help
- strict local GGUF run
- full test suite: 90 tests passed
- Docker execution with `--network none`

The pushed tag was pulled back from Docker Hub and executed three independent
times with `--network none`. Every run:

- exited successfully;
- produced exactly the columns `qid,answer`;
- preserved input qid order;
- produced only answers in `A/B/C/D`;
- required no GPU, CUDA, network, API, or external model mount.

The published manifest targets `linux/amd64`. The final digest is recorded in
`SUBMISSION.md` after the pushed image is pulled and validated.

## 8. Reproduction

```bash
docker pull viape/hackaithon-c:v9-3-final
docker run --rm --network none \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/output:/output" \
  viape/hackaithon-c:v9-3-final
```

The resulting predictions are written to `/output/pred.csv`.
