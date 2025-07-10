from dataclasses import dataclass

import numpy as np
from Layer import Layer, Layers_type, Activation_fn


@dataclass
class Parameters_dense:
    weights: np.ndarray
    biases: np.ndarray


@dataclass
class Gradients_dense:
    g_weights: np.ndarray
    g_biases: np.ndarray


@dataclass
class Cache_dense:
    Z_prev: np.ndarray
    Z_current: np.ndarray


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

        self.n_neurons_current = None
        self.input_shape = None
        self.n_neurons_post = output_shape[0]

        self.parameters = Parameters_dense(
            weights=np.array([]),
            biases=np.array([]),
        )

        self.gradients = Gradients_dense(
            g_weights=np.array([]),
            g_biases=np.array([]),
        )

        self.is_initialized = False

        self.cache = None
        self.activation_fn = activation_fn

    def initialize_parameters(self):

        if self.input_shape is None:
            raise ValueError(
                "Input shape must be defined before initializing parameters, use set_input_shape() before initialize_parameters()."
            )

        self.n_neurons_current = self.input_shape[0]

        self.parameters = Parameters_dense(
            weights=np.random.randn(self.n_neurons_current, self.n_neurons_post)
            * 0.01,  # shape (n_neurons_current, n_neurons_post)
            biases=np.zeros((1, self.n_neurons_post)),
        )

        self.gradients = Gradients_dense(
            g_weights=np.zeros((self.n_neurons_current, self.n_neurons_post)),
            g_biases=np.zeros((1, self.n_neurons_post)),
        )

        self.is_initialized = True

    def set_input_shape(self, input_shape: tuple | None):
        if input_shape is None:
            raise ValueError("Input shape cannot be None.")
        self.input_shape = input_shape

    def forward_step(self, data_in: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise ValueError("Parameters must be initialized before forward pass.")

        if data_in.shape[1] != self.n_neurons_current:
            raise ValueError(
                f"Input shape {data_in.shape[1]} does not match expected shape {self.n_neurons_current}."
            )
        if self.parameters.weights.shape[0] != data_in.shape[1]:
            raise ValueError(
                f"Weight shape {self.parameters.weights.shape[0]} does not match input shape {data_in.shape[1]}."
            )

        W = self.parameters.weights
        b = self.parameters.biases
        Z_prev = data_in

        Z_current = Z_prev @ W + b  # shape (n_samples, n_neurons_post)

        activation = self.activation_fn

        if activation == Activation_fn.RELU:
            Z_current = np.maximum(0, Z_current)
        elif activation == Activation_fn.SIGMOID:
            Z_current = 1 / (1 + np.exp(-Z_current))
        elif activation == Activation_fn.TANH:
            Z_current = np.tanh(Z_current)
        elif activation == Activation_fn.SOFTMAX:
            exp_values = np.exp(Z_current - np.max(Z_current, axis=1, keepdims=True))
            Z_current = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.cache = Cache_dense(Z_prev=Z_prev, Z_current=Z_current)
        return Z_current

    def forward(self, data_in: np.ndarray) -> np.ndarray:
        return data_in

    def backward(self, gradient_in: np.ndarray) -> np.ndarray:
        return gradient_in
