# Assessing the Probabilistic Fit of Neural Regressors

This repository contains the official implementation of "Beyond Calibration: Assessing the Probabilistic Fit of Neural Regressors via Conditional Congruence".

Our repository name, `probcal`, is a portmanteau of "problems" and "calibration", i.e. we address existing problems with using calibration to evaluate neural nets.

## Important Links

Important figures used in the paper, along with the code that generated them, can be found in [this directory](probcal/figures).

Our implementations of the probabilistic neural networks referenced in the paper can be found in [this directory](probcal/models).

Saved model weights can be found [here](weights), and synthetic dataset files can be found [here](data). Configs to reproduce the models referenced in the paper are saved in the [configs](configs) directory.

## Install Project Dependencies

```bash
conda create --name probcal python=3.10
conda activate probcal
pip install -r requirements.txt
```

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting. There will also be a flake8 check that provides warnings about various Python styling violations. These must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.

## Training models

To train a probabilistic neural network, first fill out a config (using [this config](probcal/training/sample_train_config.yaml) as a template). Then, from the terminal, run

```bash
python probcal/training/train_model.py --config path/to/your/config.yaml
```

Logs / saved model weights will be found at the locations specified in your config.

### Training on Tabular Datasets

If fitting a model on tabular data, the training script assumes the dataset will be stored locally in `.npz` files with `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test` splits. Pass a path to this `.npz` file in the `dataset` `path` key in the config (also ensure that the `dataset` `type` is set to `tabular` and the `dataset` `input_dim` key is properly specified).

### Adding New Models

All regression models should inherit from the `RegressionNN` class (found [here](probcal/models/regression_nn.py)). This base class is a `lightning` module, which allows for a lot of typical NN boilerplate code to be abstracted away. Beyond setting a few class attributes like `loss_fn` while calling the super-initializer, the only methods you need to actually write to make a new module are:

- `_forward_impl` (defines a forward pass through the network)
- `_predict_impl` (defines how to make predictions with the network, including any transformations on the output of the forward pass)
- `_sample_impl` (defines how to sample from the network's learned posterior predictive distribution for a given input)
- `_posterior_predictive_impl` (defines how to produce a posterior predictive distribution from network output)
- `_point_prediction_impl` (defines how to interpret network output as a single point prediction for a regression target)
- `_addl_test_metrics_dict` (defines any metrics beyond rmse/mae that are computed during model evaluation)
- `_update_addl_test_metrics_batch` (defines how to update additional metrics beyond rmse/mae for each test batch).

See existing model classes like `GaussianNN` (found [here](probcal/models/gaussian_nn.py)) for an example of these steps.

## Evaluating Models

To obtain evaluation metrics for a given model, first fill out a config (using [this config](probcal/evaluation/sample_eval_config.yaml) as a template).
Then, run the following command:

```bash
python probcal/evaluation/eval_model.py --config path/to/eval/config.yaml
```

Two results files will be saved to the `log_dir` you specify in your config:

- A `test_metrics.yaml` with metrics like MAE, RMSE, etc. and a summary of the probabilistic results (such as the mean CCE values for each specified trial)
- A `probabilistic_results.npz` file which can be loaded into a `ProbabilisticResults` object to see granular CCE and ECE results.

## Measuring Probabilistic Fit

Once a `RegressionNN` subclass is trained, its probabilistic fit can be measured on a dataset via the `ProbabilisticEvaluator`. Example usage:

```python
from probcal.data_modules import COCOPeopleDataModule
from probcal.enums import DatasetType
from probcal.evaluation import ProbabilisticEvaluator
from probcal.evaluation import ProbabilisticEvaluatorSettings
from probcal.models import GaussianNN


# You can customize the settings for the CCE / ECE computation.
settings = ProbabilisticEvaluatorSettings(
    dataset_type=DatasetType.IMAGE,
    cce_input_kernel="polynomial",
    cce_output_kernel="rbf",
    cce_lambda=0.1,
    cce_num_samples=1,
    ece_bins=50,
    ece_weights="frequency",
    ece_alpha=1,
    # etc.
)
evaluator = ProbabilisticEvaluator(settings)

model = GaussianNN.load_from_checkpoint("path/to/model.ckpt")

# You can use any lightning data module (preferably, the one with the dataset the model was trained on).
data_module = COCOPeopleDataModule(root_dir="data", batch_size=4, num_workers=0, persistent_workers=False)
results = evaluator(model=model, data_module=data_module)
results.save("path/to/results.npz")
```

Invoking the `ProbabilisticEvaluator`'s `__call__` method (as above) kicks off an extensive evaluation wherein CCE and ECE are computed for the specified model. This passes back an `ProbabilisticResults` object, which will contain the computed metrics and other helpful variables for further analysis.
