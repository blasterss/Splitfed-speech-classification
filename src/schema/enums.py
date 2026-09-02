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


class OptimizerType(str, Enum):
    adam = "adam"


class NoiseType(str, Enum):
    gauss = "gauss"
    laplace = "laplace"


class TrainingMode(str, Enum):
    centralized = "centralized"
    federated = "federated"
    split = "split"
    splitfed = "splitfed"


class ServerModelScope(str, Enum):
    shared = "shared"
    personalized = "personalized"


class SplitServerStrategy(str, Enum):
    """Server update ordering for a shared split model."""

    concat_v1 = "concat_v1"
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
