# Container Contract

- Read input from `/data/private_test.csv` if present; otherwise read `/data/public_test.csv`.
- Write predictions to `/output/pred.csv`.
- Output columns: `qid,answer`.
- Valid answer labels: `A`, `B`, `C`, `D`.
- The runtime must not require network access during scoring.
