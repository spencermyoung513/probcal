#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset-name> <experiment-name>"
    echo "Example: $0 coco blur"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Usage: $0 <dataset-name> <experiment-name>"
    echo "Example: $0 coco blur"
    exit 1
fi

dataset=$1
experiment_name=$2
configs_dir="configs/experiments/${dataset}"
echo "${configs_dir}"

if [ -d "$configs_dir" ]; then
    :
else
    echo "Error: '$dataset' is not a supported dataset."
    exit 1
fi

experiment_dir="${configs_dir}/${experiment_name}"

if [ -d "$experiment_dir" ]; then
    :
else
    echo "Error: '$experiment_name' is not a supported experiment."
    exit 1
fi

for model_dir in ${experiment_dir}/*; do
    model_name=$(basename "$model_dir")
    echo "Running ${experiment_name} ood exp on ${dataset} ${model_name} head"

    head_list=()
    for file in ${experiment_dir}/${model_name}/*.yaml; do
      file_name=$(basename "$file" .yaml)
      head_list+=("$file_name")
    done

    for head in "${head_list[@]}"; do
        python probcal/experiments/ood.py \
            --config ${experiment_dir}/${model_name}/${head}.yaml
    done
    echo "${experiment_name} ood exp on ${dataset} ${model_name} head done."
done
