from enum import Enum


class AcceleratorType(Enum):
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"
    auto = "auto"


class HeadType(Enum):
    GAUSSIAN = "gaussian"
    FAITHFUL_GAUSSIAN = "faithful_gaussian"
    NATURAL_GAUSSIAN = "natural_gaussian"
    POISSON = "poisson"
    NEGATIVE_BINOMIAL = "negative_binomial"
    DOUBLE_POISSON = "double_poisson"


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAM_W = "adam_w"


class LRSchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"


class BetaSchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"


class DatasetType(Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"


class ImageDatasetName(Enum):
    ROTATED_MNIST = "rotated_mnist"
    COCO_PEOPLE = "coco_people"
    AAF = "aaf"
    EVA = "eva"
    OOD_BLUR_EVA = "ood_blur_eva"
    OOD_MIXUP_EVA = "ood_mixup_eva"
    OOD_LABEL_NOISE_EVA = "ood_label_noise_eva"
    OOD_BLUR_COCO_PEOPLE = "ood_blur_coco_people"
    OOD_MIXUP_COCO_PEOPLE = "ood_mixup_coco_people"
    OOD_LABEL_NOISE_COCO_PEOPLE = "ood_label_noise_coco_people"
    FG_NET = "fg_net"


class TextDatasetName(Enum):
    READABILITY = "readability"
