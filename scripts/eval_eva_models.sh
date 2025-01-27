#!/bin/bash

# Directory containing the config files
CONFIG_DIR="configs/test/eva/"

echo "Running training with config files in $CONFIG_DIR"

for config_file in "$CONFIG_DIR"*; do
    if [ -f "$config_file" ]; then
        echo "Running training with config file: $config_file"
        python probcal/evaluation/eval_model.py --config "$config_file"
    fi
done
