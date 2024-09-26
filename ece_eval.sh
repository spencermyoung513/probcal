#!/bin/bash

echo "Beginning model evaluation."

python -m probcal.evaluation.eval_model --config configs/test/aaf/gaussian_aaf_cfg.yaml
python -m probcal.evaluation.eval_model --config configs/test/aaf/poisson_aaf_cfg.yaml
python -m probcal.evaluation.eval_model --config configs/test/aaf/double_poisson_aaf_cfg.yaml
python -m probcal.evaluation.eval_model --config configs/test/aaf/negative_binomial_aaf_cfg.yaml

echo "All models finished evaluation successfully."