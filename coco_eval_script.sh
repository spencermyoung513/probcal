# #!/bin/bash

# echo "evaluating nbinom model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_nbinom_cfg.yaml
# echo "evaluating ddpn model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_ddpn_cfg.yaml
# echo "evaluating immer model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_immer_cfg.yaml
# echo "evaluating poisson model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_poisson_cfg.yaml
echo "evaluating seitzer model..."
python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_seitzer_cfg.yaml
# echo "evaluating stirn model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_stirn_cfg.yaml
# echo "evaluating gaussian model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_gaussian_cfg.yaml
