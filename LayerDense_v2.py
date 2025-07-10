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

        self.output_shape = (
            1,
            output_shape[0],
        )  # batch size is always 1 for this implementation

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
        self.already_forwarded = False
        self.already_backwarded = False

        self.cache = None
        self.activation_fn = activation_fn

    def initialize_parameters(self):
        if self.input_shape is None:
            raise ValueError("Call set_input_shape() first.")
        _, n_in = self.input_shape
        n_out = self.output_shape[1]  # output_shape (1, n_neurons_post)
        self.parameters = Parameters_dense(
            weights=np.random.randn(n_in, n_out) * 0.01,
            biases=np.zeros((1, n_out)),
        )
        self.gradients = Gradients_dense(
            g_weights=np.zeros((n_in, n_out)),
            g_biases=np.zeros((1, n_out)),
        )
        self.is_initialized = True

    def set_input_shape(self, input_shape: tuple[int, ...] | None):
        if input_shape is None:
            raise ValueError("Input shape cannot be None.")

        # (n,) or (1, n)
        if len(input_shape) == 2 and input_shape[0] == 1:
            input_shape = (input_shape[1],)

        if len(input_shape) != 1:
            raise ValueError(f"Input shape must be (n_features,), got {input_shape}")

        self.input_shape = (1, input_shape[0])  # forma interna con batch 1
        self.n_neurons_current = input_shape[0]

    def forward_step(self, data_in: np.ndarray) -> np.ndarray:
        W = self.parameters.weights
        b = self.parameters.biases
        Z_prev = data_in

        A_current = Z_prev @ W + b  # shape (n_samples, n_neurons_post)

        activation = self.activation_fn
        Z_current = None

        if activation == Activation_fn.RELU:
            Z_current = np.maximum(0, A_current)
        elif activation == Activation_fn.SIGMOID:
            Z_current = 1 / (1 + np.exp(-A_current))
        elif activation == Activation_fn.TANH:
            Z_current = np.tanh(A_current)
        elif activation == Activation_fn.SOFTMAX:
            exp_values = np.exp(A_current - np.max(A_current, axis=1, keepdims=True))
            Z_current = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        return Z_current

    def forward(self, data_in: np.ndarray) -> np.ndarray:
        # (n_features,) o (batch, n_features)
        if data_in.ndim == 1:
            data_in = data_in.reshape(1, -1)  # batch 1
        if data_in.ndim != 2:
            raise ValueError("Expected 1D or 2D input, got shape " + str(data_in.shape))

        batch_size, n_in = data_in.shape
        if n_in != self.n_neurons_current:
            raise ValueError(f"Expected {self.n_neurons_current} features, got {n_in}")

        Z_current = self.forward_step(data_in)
        self.cache = Cache_dense(
            Z_prev=data_in,
            Z_current=Z_current,
        )

        self.already_forwarded = True
        self.already_backwarded = False
        return Z_current

    def backward_step(
        self, dZ_current: np.ndarray
    ) -> tuple[np.ndarray, Gradients_dense]:
        if self.cache is None:
            raise ValueError(
                "Cache is None. Make sure forward() is called before backward_step()."
            )

        Z_prev = self.cache.Z_prev
        Z_current = self.cache.Z_current

        m = Z_prev.shape[0]

        dA = None
        dg = None  # Gradient of the activation function

        g = self.activation_fn

        if g == Activation_fn.SIGMOID:
            s = 1 / (1 + np.exp(-Z_current))
            dg = s * (1 - s)
        elif g == Activation_fn.RELU:
            dg = np.where(Z_current > 0, 1, 0)
        elif g == Activation_fn.TANH:
            dg = 1 - np.tanh(Z_current) ** 2
        elif g == Activation_fn.SOFTMAX:
            dg = 1
        else:
            raise ValueError("Función de activación desconocida: " + g)

        dA = dZ_current * dg  # regla de la cadena dz = dl/dZ * dg/dA

        dW = Z_prev.T @ dA / m  # (n_in, m) @ (m, n_out) = (n_in, n_out)
        db = np.sum(dA, axis=0, keepdims=True) / m  # shape: (1, n_out)
        dA_prev = (
            dA @ self.parameters.weights.T
        )  # (m, n_out) @ (n_out, n_in) = (m, n_in)

        gradients = Gradients_dense(g_weights=dW, g_biases=db)
        return dA_prev, gradients

    def backward(self, gradient_in: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise ValueError("Parameters must be initialized before backward pass.")
        if not self.already_forwarded:
            raise ValueError("Forward pass must be called before backward step.")

        if gradient_in.ndim == 1:
            gradient_in = gradient_in.reshape(1, -1)
        # batch 1 (batch, n_neurons)

        dZ_prev, gradients = self.backward_step(gradient_in)
        self.gradients = gradients

        self.already_forwarded = False
        self.already_backwarded = True

        return dZ_prev

    def update_parameters(self, learning_rate: float):
        if not self.is_initialized:
            raise ValueError("Parameters must be initialized before updating.")

        if not self.already_backwarded:
            raise ValueError("Backward pass must be called before updating parameters.")

        W = self.parameters.weights
        b = self.parameters.biases

        gW = self.gradients.g_weights
        gb = self.gradients.g_biases

        # Update weights and biases with momentum
        W -= learning_rate * gW
        b -= learning_rate * gb

        # Update parameters
        self.parameters.weights = W
        self.parameters.biases = b

        self.already_backwarded = False

    def get_activation_function(self) -> Activation_fn:
        return self.activation_fn

    def get_parameters(self):
        return self.parameters

    def get_gradients(self):
        return self.gradients
