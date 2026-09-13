"""Stable enum values used by configuration models."""

from enum import Enum


class TransportType(str, Enum):
    """Transport layer type for inter-component communication."""

    queue = "queue"
    grpc = "grpc"


class DatasetType(str, Enum):
    """Supported datasets."""

    crema_d = "CREMA-D"
    ravdess = "RAVDESS"
    savee = "SAVEE"


class AggregationStrategy(str, Enum):
    """Parameter aggregation strategy on the federated server."""

    fedavg = "fedavg"
    weighted_fedavg = "weighted_fedavg"
    mergesfl_batch_weighted_v1 = "mergesfl_batch_weighted_v1"


class OptimizerType(str, Enum):
    adam = "adam"
    sgd = "sgd"


class NoiseType(str, Enum):
    gauss = "gauss"
    laplace = "laplace"


class TrainingMode(str, Enum):
    local = "local"
    centralized = "centralized"
    federated = "federated"
    split = "split"
    splitfed = "splitfed"


class WorkloadPolicy(str, Enum):
    """Client-side stopping rule for one training round."""

    max_steps_v1 = "max_steps_v1"
    full_epoch_v1 = "full_epoch_v1"
    fixed_steps_v1 = "fixed_steps_v1"


class ServerModelScope(str, Enum):
    shared = "shared"
    personalized = "personalized"


class SplitServerStrategy(str, Enum):
    """Server update ordering for a shared split model."""

    concat_v1 = "concat_v1"
    mergesfl_v1 = "mergesfl_v1"
    mergesfl_algorithm1_v1 = "mergesfl_algorithm1_v1"
    sequential_v1 = "sequential_v1"


class ExperimentProfile(str, Enum):
    smoke = "smoke"
    unit = "unit"


class FeatureType(str, Enum):
    """Types of extracted features."""

    mel = "mel"
    mfcc = "mfcc"
    rms = "rms"
    zcr = "zcr"
    contrast = "contrast"
