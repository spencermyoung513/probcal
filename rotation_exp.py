import csv
import os
import sys
from pathlib import Path
from typing import Type

import lightning as L
import torch

from probcal.enums import HeadType
from probcal.evaluation.probabilistic_evaluator import ProbabilisticEvaluator
from probcal.evaluation.probabilistic_evaluator import ProbabilisticEvaluatorSettings
from probcal.models.multi_class_nn import MultiClassNN
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_multi_class_model


def eval_model(config_path: Path):
    config = EvaluationConfig.from_yaml(config_path)
    if not config.log_dir.exists():
        os.makedirs(config.log_dir)

    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_path_or_spec,
        config.batch_size,
        rotation=config.rotation,
    )

    print(f"Using rotation: {config.rotation} on MNIST data...")

    if config.head_type == HeadType.MULTI_CLASS:
        initializer: Type[MultiClassNN] = get_multi_class_model(config, return_initializer=True)[1]
        model = initializer.load_from_checkpoint(config.model_ckpt_path)
        evaluator = L.Trainer(
            accelerator=config.accelerator_type.value,
            enable_model_summary=False,
            logger=False,
            devices=1,
            num_nodes=1,
        )
        metrics: dict = evaluator.test(model=model, datamodule=datamodule)[0]
        metrics = {k: float(v) for k, v in metrics.items()}

    cce_input_kernel = "polynomial"

    prob_eval_settings = ProbabilisticEvaluatorSettings(
        dataset_type=config.dataset_type,
        device=torch.device(config.accelerator_type.value),
        cce_num_trials=config.cce_num_trials,
        cce_input_kernel=cce_input_kernel,
        cce_output_kernel=config.cce_output_kernel,
        cce_output_kernel_eps=config.epsilon,
        cce_lambda=config.cce_lambda,
        cce_num_samples=config.cce_num_samples,
        ece_bins=config.ece_bins,
        ece_weights=config.ece_weights,
        ece_alpha=config.ece_alpha,
    )

    prob_evaluator = ProbabilisticEvaluator(settings=prob_eval_settings)
    print("Evaluating probabilistic fit...")
    results = prob_evaluator(model=model, data_module=datamodule)

    metrics.update(
        mean_cce=[float(result.mean_cce) for result in results.cce_results],
        ece=float(results.ece),
    )

    return config.rotation, metrics


def save_experiment_result(rotation, results, filename="rotation_experiment_results.csv"):
    """
    Save a single experiment result to a CSV file.

    Args:
        experiment_config (dict): Configuration parameters for the experiment
        result (float): The result of the experiment
        filename (str): Name of the CSV file to save results
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)

    # Get all configuration keys to use as columns
    headers = ["rotation", "cce", "ece", "test_acc"]

    # Combine configuration and result into one row
    row_data = {
        "rotation": rotation,
        "cce": results["mean_cce"][0],
        "ece": results["ece"],
        "test_acc": results["test_acc"],
    }

    # Write to CSV file
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        # Write headers only if file is being created for the first time
        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)


def main(directory):
    if not os.path.isdir(directory):
        print(f"The provided path '{directory}' is not a valid directory.")
        return

    # Your code to process the directory goes here
    print(f"---Processing directory: {directory}---")
    for config in os.listdir(directory):
        if config.endswith(".yaml"):
            config_path = Path(directory) / config
            rotation, results = eval_model(config_path)
            # save_experiment_result(rotation, results)
            # print(f" - Results for {config_path.stem} saved to CSV.")
    print("---Done processing directory---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rotation_exp.py <directory>")
    else:
        main(sys.argv[1])
