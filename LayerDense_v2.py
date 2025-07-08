from dataclasses import dataclass

import numpy as np
from Layer import Layer, Layers_type, Activation_fn


class LayerDense(Layer):
    def __init__(
        self,
        output_shape: tuple[int],
        activation_fn: Activation_fn,
        input_shape: tuple[int] | None = None,
        name: str | None = None,
    ):
        super().__init__(
            layer_type=Layers_type.DENSE, input_shape=input_shape, name=name
        )
        self.output_shape = output_shape
        self.parameters = {
            "weights": None,
            "biases": None,
        }
        self.cache = None
        self.activation_fn = activation_fn

    def initialize_parameters(self):
        pass

    def forward(self, data_in: np.ndarray) -> np.ndarray:
        return data_in

    def backward(self, gradient_in: np.ndarray) -> np.ndarray:
        return gradient_in
