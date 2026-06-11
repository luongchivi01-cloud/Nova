# Nova HackAIthon 2026 - Bang C CPU Submission

Offline, CPU-portable multiple-choice solver for HackAIthon 2026 Bang C -
Innovator.

## Final artifact

- Docker image: `viape/hackaithon-c:v9-3-final`
- Docker Hub: https://hub.docker.com/r/viape/hackaithon-c/tags
- Platform: `linux/amd64`
- Runtime: CPU-only `llama-cpp-python`
- Model: self-contained Qwen3.5-9B Q4_K_M GGUF
- Published digest:
  `sha256:ffcbeeb9a600f6f4820dc01868b2af6ee7d2024fd1f4a929a0a155daec8c380a`

The submitted image requires no GPU, CUDA, runtime network, external API, or
external model mount.

## Competition contract

The container reads `/data/private_test.csv` when present, otherwise
`/data/public_test.csv`, and writes `/output/pred.csv`.

Required output schema:

```csv
qid,answer
```

Every answer is constrained to exactly one of `A`, `B`, `C`, or `D`. Input
`qid` order is preserved.

## Reproduce the submitted image

Place `public_test.csv` or `private_test.csv` in `./data`, then run:

```bash
mkdir -p output
docker pull viape/hackaithon-c:v9-3-final
docker run --rm --network none \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/output:/output" \
  viape/hackaithon-c:v9-3-final
cat output/pred.csv
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force output | Out-Null
docker pull viape/hackaithon-c:v9-3-final
docker run --rm --network none `
  -v "${PWD}/data:/data:ro" `
  -v "${PWD}/output:/output" `
  viape/hackaithon-c:v9-3-final
Get-Content output/pred.csv
```

Run the complete three-pass validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_final_image.ps1
```

The public test is used only for runtime and format validation. No expected
answer sequence is asserted.

## CPU solver design

The official CPU profile uses:

1. One grammar-constrained model call for low-risk questions.
2. One direct call plus at most one verifier call for high-risk questions.
3. GGUF chat-template generation constrained to exactly `A/B/C/D`.
4. Strict failure when the model is unavailable or an answer cannot be
   produced. There is no heuristic fallback in official mode.

Permutation loops, pairwise loops, multi-call voting, RAG, network access,
CUDA, and GPU layers are disabled in the final image.

## Build from source

The published image already contains the model. To rebuild it, provide a
directory containing `qwen3.5-9b-q4_k_m.gguf` as the named Buildx context:

```bash
docker buildx build --platform linux/amd64 --push \
  -f Dockerfile.cpu-final \
  --build-context modelsrc=/path/to/gguf-directory \
  -t viape/hackaithon-c:v9-3-final .
```

The GGUF is intentionally excluded from this GitHub repository because it is
multi-gigabyte. It is included in the published Docker image.

## Method document

See [METHOD.md](METHOD.md) for model selection, solver design, dependency
decisions, and final validation results.
