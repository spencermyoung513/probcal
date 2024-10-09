#!/bin/bash
export PYTHONPATH=/home/porter/code/probcal

n=4
m=5


echo "OOD seitzer Gaussian"
for ((i=0; i<n; i++))
do
  for ((j=0; j<m; j++))
    do
    python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_mixup_coco_seitzer_${j}.yaml
    done
done



# POISSON
echo "OOD Poisson"
for ((i=0; i<n; i++))
do
  for ((j=0; j<m; j++))
    do
    echo "SKIP"
    #python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_mixup_coco_poisson_${j}.yaml
    done
done


# DDPN
echo "OOD DDPN"
for ((i=0; i<n; i++))
do
  for ((j=0; j<m; j++))
    do
    echo "SKIP"
    #python3 probcal/experiments/ood.py --cfg-path configs/experiments/ood_mixup_coco_ddpn_${j}.yaml
    done
done
