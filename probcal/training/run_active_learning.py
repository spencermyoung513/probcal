import math
from argparse import ArgumentParser
from argparse import Namespace

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger

from probcal.utils.configs import ActiveLearningConfig
from probcal.utils.configs import TrainingConfig
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_chkp_callbacks
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model


# TODO: Make these configurable.
seed_size = 1000


def main(al_config: ActiveLearningConfig, trn_config: TrainingConfig, seed: int = 1998):
    # TODO: Add warnings about:
    # if more than 1 trial specified in trn_config, this will be ignored
    # if random seed specified in trn_config, this will be ignored (is there another way to handle this?)
    datamodule = get_datamodule(
        trn_config.dataset_type,
        trn_config.dataset_path_or_spec,
        trn_config.batch_size,
        trn_config.num_workers,
    )
    datamodule.setup("train")
    full_train = datamodule.train
    if full_train is None:
        raise RuntimeError("Expected datamodule.train to be initialized after calling `setup`.")
    init_seed_generator = torch.Generator().manual_seed(seed)
    init_train_indices = torch.multinomial(
        input=torch.arange(len(full_train)),
        num_samples=al_config.init_n_train,
        generator=init_seed_generator,
    )
    # TODO: Set datamodule train indices (will be used even if we re-init train). Will require slight rewrite of datamodule.
    # Or we make a wrapper class and re-instantiate (ActiveLearningDataModule initializes with one argument, a ProbcalDataModule).f

    for i in range(al_config.num_cycles):
        for j in range(num_al_cycles):
            fix_random_seed(config.random_seed)
            datamodule = get_datamodule(
                config.dataset_type,
                config.dataset_path_or_spec,
                config.batch_size,
            )
            # TODO: continue.
            torch.multinomial(len(datamodule.train))

            model = get_model(config)
            chkp_dir = config.chkp_dir / config.experiment_name / f"version_{i}"
            chkp_callbacks = get_chkp_callbacks(chkp_dir, config.chkp_freq)
            logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)

            trainer = L.Trainer(
                accelerator=config.accelerator_type.value,
                min_epochs=config.num_epochs,
                max_epochs=config.num_epochs,
                log_every_n_steps=5,
                check_val_every_n_epoch=math.ceil(config.num_epochs / 200),
                enable_model_summary=False,
                callbacks=chkp_callbacks,
                logger=logger,
                precision=config.precision,
            )
            trainer.fit(model=model, datamodule=datamodule)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(TrainingConfig.from_yaml(args.config))
