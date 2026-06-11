# Submission Artifacts

## Reproducibility artifacts

- Docker Hub: https://hub.docker.com/r/viape/hackaithon-c/tags
- Final image: `viape/hackaithon-c:v9-3-final`
- GitHub: https://github.com/luongchivi01-cloud/Nova
- Method: https://github.com/luongchivi01-cloud/Nova/blob/main/METHOD.md
- Third-party notices:
  https://github.com/luongchivi01-cloud/Nova/blob/main/THIRD_PARTY_NOTICES.md

## Final image identity

```text
Platform: linux/amd64
Digest: sha256:66472c3724c6be5eecf2acf78741c76e42ad63587611cf01f670da111b3af34e
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
- [x] Third-party model and runtime notices are included.
- [x] Full Round 1 prediction CSV has passed strict local validation.

## Notes for organizers

The public test is used as unlabeled inference input and for runtime and CSV
format validation. The submission contains no supplied labels, fixed
public-answer assertion, or hard-coded answer sequence.
