#!/bin/bash

echo "Beginning model evaluation."

python -m probcal.experiments.aaf_2e --config configs/train/aaf/double_poisson_aaf_cfg.yaml
python -m probcal.experiments.aaf_2e --config configs/train/aaf/gaussian_aaf_cfg.yaml
python -m probcal.experiments.aaf_2e --config configs/train/aaf/negative_binomial_aaf_cfg.yaml
python -m probcal.experiments.aaf_2e --config configs/train/aaf/poisson_aaf_cfg.yaml

echo "All models finished evaluation successfully."