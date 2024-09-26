from argparse import ArgumentParser, Namespace
from probcal.enums import DatasetType
from probcal.evaluation import CalibrationEvaluator
from probcal.evaluation import CalibrationEvaluatorSettings

from probcal.utils.configs import TrainingConfig
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model

import os
from pathlib import Path

def main(config_path: Path):

    config = EvaluationConfig.from_yaml(config_path)
    if not config.log_dir.exists():
        os.makedirs(config.log_dir)
    
    # Set Variables
    model_name = config.experiment_name
    checkpoint = config.model_ckpt_path
    path_to_model = os.path.join('chkp', model_name, 'version_0', f'{checkpoint}.ckpt')

    output_dir = f"results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    
    # Set up CalibrationEvaluator
    settings = CalibrationEvaluatorSettings(
        dataset_type=DatasetType.IMAGE,
        mcmd_input_kernel="polynomial",
        mcmd_output_kernel="rbf",
        mcmd_lmbda=0.1,
        mcmd_num_samples=5,
        ece_bins=50,
        ece_weights="frequency",
        ece_alpha=1,
    )
    evaluator = CalibrationEvaluator(settings)

    # Get model, data, and load from checkpoint
    fix_random_seed(config.random_seed)
    _, initializer = get_model(config, return_initializer=True)
    
    model = initializer.load_from_checkpoint(path_to_model)
    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_path_or_spec,
        config.batch_size,
    )

    # Run and save the evaluator
    calibration_results = evaluator(model=model, data_module=datamodule)
    calibration_results.save(output_dir + "/" + checkpoint + ".npz")

    plot = evaluator.plot_mcmd_results(calibration_results, title=f"{model_name}: {checkpoint}")
    plot.savefig(output_dir +  "/" + checkpoint + ".png")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(config_path=Path(args.config))