from argparse import ArgumentParser, Namespace
from probcal.evaluation import CalibrationResults
from probcal.utils.configs import EvaluationConfig

def main(config: EvaluationConfig):
    calibration_results_file_path = "logs/eval/" + config.experiment_name + "/calibration_results.npz"

    calibration_results = CalibrationResults.load(calibration_results_file_path)

    print("Mean MCMD:", calibration_results.mean_mcmd)
    print("ECE:", calibration_results.ece)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(EvaluationConfig.from_yaml(args.config))
