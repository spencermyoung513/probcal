echo "Evaluating DDPN"
python3 probcal/evaluation/eval_model.py --config configs/test/coco-people/ddpn.yaml
echo "evaluating immer"
python3 probcal/evaluation/eval_model.py --config configs/test/coco-people/immer.yaml
echo "evaluating nbinom"
python3 probcal/evaluation/eval_model.py --config configs/test/coco-people/nbinom.yaml
echo "evaluating poisson"
python3 probcal/evaluation/eval_model.py --config configs/test/coco-people/poisson.yaml
echo "evaluating seitzer"
python3 probcal/evaluation/eval_model.py --config configs/test/coco-people/seitzer.yaml
echo "evaluating stirn"
python3 probcal/evaluation/eval_model.py --config configs/test/coco-people/stirn.yaml
