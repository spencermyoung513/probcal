#!/bin/bash

# Check if directory path is provided
if [ $# -eq 0 ]; then
    echo "Please provide the directory path as an argument."
    exit 1
fi

# Directory path
dir_path="$1"

# Check if directory exists
if [ ! -d "$dir_path" ]; then
    echo "Directory does not exist: $dir_path"
    exit 1
fi

# Python command (replace with your actual command)
python_cmd="python probcal/evaluation/eval_model.py --config "

# Loop through all files in the directory
for file in "$dir_path"/*.yaml; do
    # Check if file exists (this is necessary to handle the case of no .conf files)
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        $python_cmd "$file"
    fi
done

echo "All config files processed."
