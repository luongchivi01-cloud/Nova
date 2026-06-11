# Round 1 Submission

## Required links

- Docker Hub: https://hub.docker.com/r/viape/hackaithon-c/tags
- Final image: `viape/hackaithon-c:v9-3-final`
- GitHub: https://github.com/luongchivi01-cloud/Nova
- Method document: https://github.com/luongchivi01-cloud/Nova/blob/main/METHOD.md

## Final image identity

```text
Platform: linux/amd64
Digest: sha256:ffcbeeb9a600f6f4820dc01868b2af6ee7d2024fd1f4a929a0a155daec8c380a
```

## Submission checklist

- [x] Docker image is self-contained and publicly pullable.
- [x] Runtime works with `--network none`.
- [x] Runtime requires CPU only and forces `N_GPU_LAYERS=0`.
- [x] Runtime reads `/data/public_test.csv` or `/data/private_test.csv`.
- [x] Runtime writes `/output/pred.csv` with exactly `qid,answer`.
- [x] All answers are constrained to `A/B/C/D`.
- [x] GitHub contains source code and reproduction instructions.
- [x] Method document describes the final submitted runtime.
- [ ] Team registration and links submitted at http://hackaithon.vsds.vn.

## Notes for organizers

The public test is used only as a smoke test for runtime and CSV format. The
submission contains no fixed public-answer assertion or answer sequence.
