import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Type

import lightning as L
import torch
import yaml

from probcal.enums import DatasetType
from probcal.enums import HeadType
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.probabilistic_evaluator import ProbabilisticEvaluator
from probcal.evaluation.probabilistic_evaluator import ProbabilisticEvaluatorSettings
from probcal.models.multi_class_nn import MultiClassNN
from probcal.models.regression_nn import RegressionNN
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model
from probcal.utils.experiment_utils import get_multi_class_model


def main(config_path: Path):

    config = EvaluationConfig.from_yaml(config_path)
    if not config.log_dir.exists():
        os.makedirs(config.log_dir)

    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_path_or_spec,
        config.batch_size,
    )

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
    else:
        initializer: Type[RegressionNN] = get_model(config, return_initializer=True)[1]
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

    if config.dataset_type == DatasetType.TABULAR and config.input_dim == 1:
        x_vals = torch.cat([x for x, _ in datamodule.test_dataloader()], dim=0)
        gamma = (1 / (2 * x_vals.var())).item()
        cce_input_kernel = partial(rbf_kernel, gamma=gamma)
    else:
        cce_input_kernel = "polynomial"

    prob_eval_settings = ProbabilisticEvaluatorSettings(
        dataset_type=config.dataset_type,
        device=torch.device(config.accelerator_type.value),
        cce_num_trials=config.cce_num_trials,
        cce_input_kernel=cce_input_kernel,
        cce_output_kernel=config.cce_output_kernel,
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
    with open(config.log_dir / "test_metrics.yaml", "w") as f:
        yaml.safe_dump(metrics, f)
    results.save(config.log_dir / "probabilistic_results.npz")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to evaluation config.yaml.")
    args = parser.parse_args()
    main(config_path=Path(args.config))
