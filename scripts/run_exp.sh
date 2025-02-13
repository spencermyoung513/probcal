#!/bin/bash

# Record the start time of the script
start_time=$(date +%s)

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset-name> <experiment-name> <num-runs>"
    echo "Example: $0 coco blur 5"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Usage: $0 <dataset-name> <experiment-name> <num-runs>"
    echo "Example: $0 coco blur 5"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Usage: $0 <dataset-name> <experiment-name> <num-runs>"
    echo "Example: $0 coco blur 5"
    exit 1
fi

if [ -n "$4" ] && [ "$4" != "ignore" ]; then
    echo "Usage: $0 <dataset-name> <experiment-name> <num-runs> [ignore]"
    echo "Example: $0 coco blur 5 ignore"
    exit 1
fi

if [ -n "$4" ] && [ "$4" == "ignore" ]; then
    echo "=> Ignoring python warnings."
    export PYTHONWARNINGS="ignore"
fi

dataset=$1
experiment_name=$2
num_runs=$3
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

for ((i=0; i<num_runs; i++)); do
    # Record the start time for each run
    run_start_time=$(date +%s)

    echo "=> Running ${experiment_name} exp on ${dataset} run ${i}"
    for model_dir in ${experiment_dir}/*; do
        model_name=$(basename "$model_dir")
        if [ "$model_name" == "gaussian" ]; then
            echo "Skipping ${experiment_name} ood exp on ${dataset} ${model_name} head"
            continue
        fi
        echo "=> Running ${experiment_name} ood exp on ${dataset} ${model_name} head"

        head_list=()
        for file in ${experiment_dir}/${model_name}/*.yaml; do
        file_name=$(basename "$file" .yaml)
        head_list+=("$file_name")
        done

        for head in "${head_list[@]}"; do
            python probcal/experiments/ood.py \
                --config ${experiment_dir}/${model_name}/${head}.yaml
        done
        echo "=> ${experiment_name} ood exp on ${dataset} ${model_name} head done."
    done

    # Calculate and display time taken for this run
    run_end_time=$(date +%s)
    run_duration=$((run_end_time - run_start_time))
    echo "=> Run ${i} completed in ${run_duration} seconds"
done

# Calculate and display total execution time
end_time=$(date +%s)
total_duration=$((end_time - start_time))

# Convert seconds to hours, minutes, and seconds for better readability
hours=$((total_duration / 3600))
minutes=$(((total_duration % 3600) / 60))
seconds=$((total_duration % 60))

echo "----------------------------------------"
echo "Total execution time:"
echo "  ${hours} hours, ${minutes} minutes, ${seconds} seconds"
echo "  (${total_duration} seconds total)"
echo "----------------------------------------"
