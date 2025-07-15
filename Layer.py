from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np


class LAYER_TYPE(Enum):
    CONVOLUTIONAL = "convolutional"
    POOLING = "pooling"
    FLATTEN = "flatten"
    DENSE = "dense"


class ACTIVATION_FN(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


class OPTIMIZER(Enum):
    NO_OPTIMIZER = "no_optimizer"
    MOMENTUM = "momentum"


@dataclass
class OptimizerConfig:
    learning_rate: float
    type: OPTIMIZER = OPTIMIZER.NO_OPTIMIZER


@dataclass
class MomentumConfig(OptimizerConfig):
    momentum: float = 0.9
    type: OPTIMIZER = OPTIMIZER.MOMENTUM


class Layer(ABC):
    def __init__(
        self,
        layer_type: LAYER_TYPE,
        input_shape: tuple[int, ...] | None = None,
        name: str | None = None,
    ):
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = None
        self.name = name

    def set_input_shape(self, input_shape: tuple | None):

        self.input_shape = input_shape

    def set_output_shape(self, output_shape: tuple | None):
        self.output_shape = output_shape

    def get_input_shape(self) -> tuple | None:
        return self.input_shape

    def get_output_shape(self) -> tuple | None:
        return self.output_shape

    def get_name(self):
        return self.name

    def get_layer_type(self):
        return self.layer_type

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError("This method should be overridden in subclasses")

    @abstractmethod
    def get_gradients(self):
        raise NotImplementedError("This method should be overridden in subclasses")

    @abstractmethod
    def get_activation_function(self) -> ACTIVATION_FN | None:
        raise NotImplementedError("This method should be overridden in subclasses")

    @abstractmethod
    def initialize_parameters(self):
        # This method should be overridden in subclasses to initialize parameters specific to the layer type
        raise NotImplementedError("This method should be overridden in subclasses")

    @abstractmethod
    def update_parameters(
        self, optimizerConfig: OptimizerConfig | MomentumConfig
    ):  # TODO: This is not scalable, IMPORTANT to change
        # This method should be overridden in subclasses to update parameters based on gradients
        raise NotImplementedError("This method should be overridden in subclasses")

    @abstractmethod
    def forward(self, data_in: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden in subclasses")

    @abstractmethod
    def backward(self, gradient_in: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden in subclasses")

    def __repr__(self):
        return f"Layer(name={self.name}, type={self.layer_type}, input_shape={self.input_shape}, output_shape={self.output_shape})"
