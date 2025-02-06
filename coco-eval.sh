echo "evaluating seitzer"
python3 probcal/evaluation/eval_model.py --config configs/test/coco-people/seitzer.yaml
echo "Evaluating DDPN"
python3 probcal/evaluation/eval_model.py --config configs/test/coco-people/ddpn.yaml
