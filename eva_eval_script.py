import argparse
import csv
import os

import yaml

from probcal.data_modules import EVADataModule
from probcal.enums import DatasetType
from probcal.evaluation import CalibrationEvaluator
from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import get_model


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main(config_path):
    config = load_config(config_path)
    name = config["experiment_name"]

    # You can customize the settings for the MCMD / ECE computation.
    settings = CalibrationEvaluatorSettings(
        dataset_type=DatasetType.IMAGE,
        mcmd_input_kernel="polynomial",
        mcmd_output_kernel="rbf",
        mcmd_lambda=0.1,
        mcmd_num_samples=5,
        ece_bins=50,
        ece_weights="frequency",
        ece_alpha=1,
    )
    evaluator = CalibrationEvaluator(settings)

    model_chpks = ["best_loss.ckpt", "best_mae.ckpt", "last.ckpt"]

    # You can use any lightning data module (preferably, the one with the dataset the model was trained on).
    data_module = EVADataModule(
        root_dir="data/eva", batch_size=4, num_workers=0, persistent_workers=False
    )

    results = []

    for chpk in model_chpks:
        print(f"plotting model from chpk:{chpk}")
        model_cfg = EvaluationConfig.from_yaml(config_path)
        model, intializer = get_model(model_cfg, return_initializer=True)

        model = intializer.load_from_checkpoint(f"chkp/{name}/version_0/{chpk}")

        if not os.path.exists(f"results/{name}"):
            os.makedirs(f"results/{name}")

        if not os.path.exists(f"results/{name}/calibration_results"):
            os.makedirs(f"results/{name}/calibration_results")

        if not os.path.exists(f"results/{name}/plots"):
            os.makedirs(f"results/{name}/plots")

        calibration_results = evaluator(model=model, data_module=data_module)
        calibration_results.save(
            f"results/{name}/calibration_results/{chpk.split('.')[0]}_calibration_results.npz"
        )

        fig = evaluator.plot_mcmd_results(calibration_results)
        fig.savefig(f"results/{name}/plots/{chpk.split('.')[0]}.png")

        results.append(
            {
                "model_chpk": chpk,
                "mean_mcmd": calibration_results.mean_mcmd,
                "ece": calibration_results.ece,
            }
        )

    # Write results to CSV file
    output_file = f"results/{name}/calibration_summary.csv"

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["model_chpk", "mean_mcmd", "ece"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration evaluation with config file.")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config)
