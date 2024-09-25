from probcal.data_modules import EVADataModule
from probcal.enums import DatasetType
from probcal.evaluation import CalibrationEvaluator
from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.models import GaussianNN


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

model = GaussianNN.load_from_checkpoint("chkp/eva_gaussian/version_0/best_mae.ckpt")

# You can use any lightning data module (preferably, the one with the dataset the model was trained on).
data_module = EVADataModule(
    root_dir="data/eva", batch_size=4, num_workers=0, persistent_workers=False
)
calibration_results = evaluator(model=model, data_module=data_module)
calibration_results.save("eva_gaussian_calibration_results.npz")
