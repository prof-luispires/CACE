# CACE — Confidence-Aware Cooperative Experts

Research implementation and Raspberry Pi 5 benchmarking utilities for the CACE multivariate IoT time-series anomaly-detection framework.

## Repository structure

- `src/` — detector implementation used by the framework.
- `config/` — frozen Expert and Coordinator configurations used for reproducibility.
- `benchmark/rpi5/` — headless Raspberry Pi 5 benchmark scripts.
- `docs/` — notes for reproducibility and data preparation.

## Raspberry Pi 5 benchmark

The RP5 benchmark supports:

- quick validation run;
- standard benchmark;
- full 135-case benchmark;
- inference-only and end-to-end timing;
- latency and throughput;
- process memory and CPU utilization;
- temperature and thermal-throttling monitoring;
- automatic system-information capture.

The benchmark is designed to use a frozen CACE configuration. Test data are evaluation-only and must not be used for parameter tuning.

See `benchmark/rpi5/README.md` for execution instructions.

## Dataset

The canonical experimental CSV archive is intentionally **not included** in this Git-ready package because it is large. Place the canonical dataset archive in the benchmark working directory when reproducing the experiments. See `docs/DATA.md`.

## Reproducibility

The configuration CSV files are included so that the RP5 deployment uses the same frozen Expert and Coordinator configuration as the main experimental pipeline.

## Citation

If this repository accompanies a published article, please cite the final publication. Bibliographic information can be added here after publication.
