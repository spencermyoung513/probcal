# #!/bin/bash
# # num_loops=5

# echo "evaluating nbinom model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_nbinom_cfg.yaml
# echo "evaluating ddpn model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_ddpn_cfg.yaml
# echo "evaluating immer model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_immer_cfg.yaml
# echo "evaluating poisson model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_poisson_cfg.yaml
# echo "evaluating seitzer model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_seitzer_cfg.yaml
# echo "evaluating stirn model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_stirn_cfg.yaml
echo "evaluating gaussian model..."
python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_gaussian_cfg.yaml

# if [ $# -eq 1 ]; then
#     # Check if the argument is a positive integer
#     if [[ $1 =~ ^[0-9]+$ ]] && [ $1 -gt 0 ]; then
#         num_loops=$1
#     else
#         echo "Error: Please provide a positive integer for the number of loops."
#         exit 1
#     fi
# fi

# echo "evaluating double possion model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_ddpn_cfg.yaml
#     python eval_script.py --results logs/coco/ddpn_results/calibration_results.npz --data logs/coco/ddpn_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."

# echo "evaluating immer model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_immer_cfg.yaml
#     python eval_script.py --results logs/coco/immer_results/calibration_results.npz --data logs/coco/immer_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."

# echo "evaluating negative binomial model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_nbinom_cfg.yaml
#     python eval_script.py --results logs/coco/nbinom_results/calibration_results.npz --data logs/coco/nbinom_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."

# echo "evaluating poisson model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_poisson_cfg.yaml
#     python eval_script.py --results logs/coco/poisson_results/calibration_results.npz --data logs/coco/poisson_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."

# echo "evaluating seitzer model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_seitzer_cfg.yaml
#     python eval_script.py --results logs/coco/seitzer_results/calibration_results.npz --data logs/coco/seitzer_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."

# echo "evaluating stirn model..."

# for i in $(seq 1 $num_loops)
# do
#     echo "MCMD $i of $num_loops"
#     python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_stirn_cfg.yaml
#     python eval_script.py --results logs/coco/stirn_results/calibration_results.npz --data logs/coco/stirn_results/mcmd_vals.txt
#     echo "Completed MCMD $i"
#     echo "-------------------"
# done

# echo "All $num_loops MCMD completed."
