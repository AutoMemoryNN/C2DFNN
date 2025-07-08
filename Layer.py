from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Layers_type(Enum):
    CONVOLUTIONAL = "convolutional"
    FLATTEN = "flatten"
    DENSE = "dense"


class Activation_fn(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


class Layer(ABC):
    def __init__(
        self,
        layer_type: Layers_type,
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
    def initialize_parameters(self):
        # This method should be overridden in subclasses to initialize parameters specific to the layer type
        raise NotImplementedError("This method should be overridden in subclasses")

    @abstractmethod
    def forward(self, data_in: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden in subclasses")

    @abstractmethod
    def backward(self, gradient_in: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden in subclasses")

    def __repr__(self):
        return f"Layer(name={self.name}, type={self.layer_type}, input_shape={self.input_shape}, output_shape={self.output_shape})"
