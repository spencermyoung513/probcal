from probcal.data_modules import COCOPeopleDataModule
from probcal.data_modules.eva_datamodule import EVADataModule
from probcal.enums import DatasetType
from probcal.evaluation import CalibrationEvaluator
from probcal.evaluation import CalibrationEvaluatorSettings
from probcal.models import GaussianNN


# You can customize the settings for the MCMD / ECE computation.
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

model = GaussianNN.load_from_checkpoint("chkp/eva/eva_model.ckpt") #TODO: modify path

data_module = EVADataModule(root_dir="data", batch_size=4, num_workers=0, persistent_workers=False)
calibration_results = evaluator(model=model, data_module=data_module)
calibration_results.save("chkp/eva/calibration/results.npz") #TODO: modify path