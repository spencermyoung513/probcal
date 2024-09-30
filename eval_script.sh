#!/bin/bash
num_loops=5

if [ $# -eq 1 ]; then
    # Check if the argument is a positive integer
    if [[ $1 =~ ^[0-9]+$ ]] && [ $1 -gt 0 ]; then
        num_loops=$1
    else
        echo "Error: Please provide a positive integer for the number of loops."
        exit 1
    fi
fi

# echo "evaluating faithful gaussian model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/eva_faithful_gaussian_eval_cfg.yaml
#     python eval_script.py --results logs/faithful_gaussian_results/calibration_results.npz --data logs/faithful_gaussian_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."

# echo "evaluating natural gaussian model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/eva_natural_gaussian_eval_cfg.yaml
#     python eval_script.py --results logs/natural_gaussian_results/calibration_results.npz --data logs/natural_gaussian_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."

# echo "evaluating gaussian beta_scheduler model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/eva_gaussian_beta_scheduler_eval_cfg.yaml
#     python eval_script.py --results logs/gaussian_beta_scheduler_results/calibration_results.npz --data logs/gaussian_beta_scheduler_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."

echo "evaluating gaussian model..."

for i in $(seq 1 $num_loops)
do
    echo "MCMD $i of $num_loops"
    python probcal/evaluation/eval_model.py --config configs/eval/eva_gaussian_eval_cfg.yaml
    python eval_script.py --results logs/gaussian_results/calibration_results.npz --data logs/gaussian_results/mcmd_vals.txt
    echo "Completed MCMD $i"
    echo "-------------------"
done

echo "All $num_loops MCMD completed."
