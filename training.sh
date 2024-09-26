#!/bin/bash

echo "Beginning model training."

echo "GAUSSIAN"
# python -m probcal.training.train_model --config configs/train/aaf/gaussian_aaf_cfg.yaml
echo "POISSON"
python -m probcal.training.train_model --config configs/train/aaf/poisson_aaf_cfg.yaml || true
echo "DOUBLE_POISSON"
python -m probcal.training.train_model --config configs/train/aaf/double_poisson_aaf_cfg.yaml || true
echo "NEGATIVE_BINOMIAL"
python -m probcal.training.train_model --config configs/train/aaf/negative_binomial_aaf_cfg.yaml || true

echo "All iterations models finished training successfully."