from dataclasses import dataclass
import numpy as np


class Layer:
    def __init__(self, layer_type: str, input_shape: np.ndarray, name: str):
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = None
        self.name = name

    def set_input_shape(self, input_shape: np.ndarray | None):

        self.input_shape = input_shape

    def set_output_shape(self, output_shape: np.ndarray | None):
        self.output_shape = output_shape

    def get_input_shape(self) -> np.ndarray | None:
        return self.input_shape

    def get_output_shape(self) -> np.ndarray | None:
        return self.output_shape

    def get_name(self):
        return self.name

    def get_layer_type(self):
        return self.layer_type

    def initialize_parameters(self):
        # This method should be overridden in subclasses to initialize parameters specific to the layer type
        raise NotImplementedError("This method should be overridden in subclasses")

    def __repr__(self):
        return f"Layer(name={self.name}, type={self.layer_type}, input_shape={self.input_shape}, output_shape={self.output_shape})"
