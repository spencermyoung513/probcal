# #!/bin/bash

#Download the files from gsutil
# cd chkp/coco/ddpn
# gsutil -m cp gs://dai-ultra-research-public/probcal/coco-people/ddpn/best_loss.ckpt .
# cd ../nbinom
# gsutil -m cp gs://dai-ultra-research-public/probcal/coco-people/nbinom/best_loss.ckpt .
# cd ../poisson
# gsutil -m cp gs://dai-ultra-research-public/probcal/coco-people/poisson/best_loss.ckpt .
# cd ../immer
# gsutil -m cp gs://dai-ultra-research-public/probcal/coco-people/immer/best_loss.ckpt .
# cd ../seitzer
# gsutil -m cp gs://dai-ultra-research-public/probcal/coco-people/seitzer/best_loss.ckpt .
# cd ../stirn
# gsutil -m cp gs://dai-ultra-research-public/probcal/coco-people/stirn/best_loss.ckpt .
# cd ../gaussian
# gsutil -m cp gs://dai-ultra-research-public/probcal/coco-people/gaussian/best_loss.ckpt .
# cd ../../..


echo "evaluating nbinom model..."
python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_nbinom_cfg.yaml
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
# echo "evaluating gaussian model..."
# python probcal/evaluation/eval_model.py --config configs/eval/coco/coco_gaussian_cfg.yaml
