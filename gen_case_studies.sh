#!/bin/bash

dataset=$1

if [ -z "$dataset" ]; then
    echo "Error: dataset argument is required"
    exit 1
fi

export PYTHONWARNINGS="ignore"

python probcal/figures/generate_case_study.py --dataset-name $dataset --dataset-type image --head immer
python probcal/figures/generate_case_study.py --dataset-name $dataset --dataset-type image --head seitzer
python probcal/figures/generate_case_study.py --dataset-name $dataset --dataset-type image --head stirn
python probcal/figures/generate_case_study.py --dataset-name $dataset --dataset-type image --head gaussian
python probcal/figures/generate_case_study.py --dataset-name $dataset --dataset-type image --head poisson
python probcal/figures/generate_case_study.py --dataset-name $dataset --dataset-type image --head ddpn
python probcal/figures/generate_case_study.py --dataset-name $dataset --dataset-type image --head nbinom
