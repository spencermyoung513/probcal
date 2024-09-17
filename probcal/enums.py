from enum import Enum


class AcceleratorType(Enum):
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"
    auto = "auto"


class HeadType(Enum):
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    NEGATIVE_BINOMIAL = "nbinom"
    MULTI_CLASS = "multi_class"


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAM_W = "adam_w"


class LRSchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"


class DatasetType(Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"

class ImageDatasetName(Enum):
    MNIST = "mnist"
    COCO_PEOPLE = "coco_people"
    CIFAR10 = 'cifar_10'
    CIFAR100 = 'cifar_100'
    AAF = 'aaf'

class TextDatasetName(Enum):
    REVIEWS = "reviews"
