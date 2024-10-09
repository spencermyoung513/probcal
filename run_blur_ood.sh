#!/bin/bash
export PYTHONPATH=/home/porter/code/probcal

n=1
m=5
# GAUSSIAN
echo "No Blur - Seitzer Gaussian Baseline"
for ((i=0; i<n; i++))
do
  python3 probcal/experiments/ood.py --cfg-path configs/experiments/coco_seitzer.yaml
done

echo "OOD Seitzer Gaussian"
for ((i=0; i<n; i++))
do
  for ((j=0; j<m; j++))
    do
    python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_seitzer_${j}.yaml
    done
done



# POISSON
echo "No Blur - Poisson Baseline$"
for ((i=0; i<n; i++))
do
  echo "SKIP"
  #python3 probcal/experiments/ood.py --cfg-path configs/experiments/coco_poisson.yaml
done

echo "OOD Poisson"
for ((i=0; i<n; i++))
do
  for ((j=0; j<m; j++))
    do
    echo "SKIP"
    #python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_poisson_${j}.yaml
    done
done


# DDPN
echo "No Blur - DDPN Baseline"
for ((i=0; i<n; i++))
do
  echo "SKIP"
  #python3 probcal/experiments/ood.py --cfg-path configs/experiments/coco_ddpn.yaml
done

echo "OOD POISSON"
for ((i=0; i<n; i++))
do
  for ((j=0; j<m; j++))
    do
    echo "SKIP"
    #python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_blur_coco_ddpn_${j}.yaml
    done
done
