echo "training the faithful guassian model..."
python probcal/training/train_model.py --config configs/train/eva/eva_faithful_gaussian_cfg.yaml
echo "evaluating faithful gaussian model..."
python eva_eval_script.py --config configs/eval/eva_faithful_gaussian_eval_cfg.yaml
echo "training and eval for the faithful gaussian model done!"

echo "training the beta scheduler guassian model..."
python probcal/training/train_model.py --config configs/train/eva/eva_gaussian_beta_scheduler_cfg.yaml
echo "evaluating beta scheduler gaussian model..."
python eva_eval_script.py --config configs/eval/eva_gaussian_beta_scheduler_eval_cfg.yaml
echo "training and eval for the beta scheduler gaussian model done!"

echo "training the natural guassian model..."
python probcal/training/train_model.py --config configs/train/eva/eva_natural_gaussian_cfg.yaml
echo "evaluating natural gaussian model..."
python eva_eval_script.py --config configs/eval/eva_natural_gaussian_eval_cfg.yaml
echo "training and eval for the natural gaussian model done!"
