import os

from probcal.data_modules import EVADataModule
from probcal.enums import DatasetType
from probcal.evaluation import CalibrationEvaluator
from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import get_model


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

ece_vals = []
mcmd_vals = []

for chpk in model_chpks:
    print(f"plotting model from chpk:{chpk}")
    # model = GaussianNN.load_from_checkpoint(f"chkp/eva_gaussian/version_0/{chpk}")
    # instantiate model
    model_cfg = EvaluationConfig.from_yaml("configs/eval/eva_gaussian_eval_cfg.yaml")
    model, intializer = get_model(model_cfg, return_initializer=True)

    model = intializer.load_from_checkpoint(f"chkp/eva_gaussian/version_0/{chpk}")

    calibration_results = evaluator(model=model, data_module=data_module)

    if not os.path.isfile(
        f"results/calibration_results/eva_gaussian_{chpk.split('.')[0]}_calibration_results.npz"
    ):
        calibration_results.save(f"eva_gaussian_{chpk.split('.')[0]}_calibration_results.npz")

        fig = evaluator.plot_mcmd_results(calibration_results)
        fig.savefig(f"results/plots/{chpk.split('.')[0]}.png")

    ece_vals.append(calibration_results.ece)
    mcmd_vals.append(calibration_results.mean_mcmd)

for i in range(len(model_chpks)):
    print(f"model_chpk: {model_chpks[i]} mean_mcmd: {mcmd_vals[i]} ece: {ece_vals[i]}")
