from probcal.models import GaussianNN
from probcal.data_modules import AAFDataModule
from probcal.evaluation import CalibrationEvaluatorSettings, CalibrationEvaluator
from probcal.enums import DatasetType

model = GaussianNN.load_from_checkpoint("chkp/aaf/version_0/last.ckpt")
data_module = AAFDataModule('data',32, 32, 32)

settings = CalibrationEvaluatorSettings()
evaluator = CalibrationEvaluator(settings)

results = evaluator(model=model,data_module=data_module)

results.save('aaf_gaussian_results.npz')