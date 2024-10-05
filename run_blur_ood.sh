#!/bin/bash
export PYTHONPATH=/Users/porterjenkins/code/probcal

n=5
# GAUSSIAN
echo "GAUSSIAN baseline $i..."
for ((i=1; i<=n; i++))
do
  echo "Running iteration $i..."
  python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_gaussian.yaml
done

echo "GAUSSIAN blur $i..."
for ((i=0; i<=n; i++))
do
  echo "Running iteration $i..."
  python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_gaussian_${i}.yaml
done

echo "Completed $n iterations."

# POISSON
echo "POISSON baseline $i..."
for ((i=1; i<=n; i++))
do
  echo "Running iteration $i..."
  python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_poisson.yaml
done

echo "GAUSSIAN blur $i..."
for ((i=0; i<=n; i++))
do
  echo "Running iteration $i..."
  python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_poisson_${i}.yaml
done

echo "Completed $n iterations."

# DDPN
echo "DDPN baseline $i..."
for ((i=1; i<=n; i++))
do
  echo "Running iteration $i..."
  python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_ddpn.yaml
done

echo "GAUSSIAN blur $i..."
for ((i=0; i<=n; i++))
do
  echo "Running iteration $i..."
  python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_ddpn_${i}.yaml
done

echo "Completed $n iterations."
