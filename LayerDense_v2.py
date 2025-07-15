from dataclasses import dataclass

import numpy as np
from Layer import Layer, LAYER_TYPE, ACTIVATION_FN, MomentumConfig, OptimizerConfig


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
        activation_fn: ACTIVATION_FN,
        input_shape: tuple[int] | None = None,
        name: str | None = None,
    ):
        super().__init__(
            layer_type=LAYER_TYPE.DENSE, input_shape=input_shape, name=name
        )

        self.output_shape = (
            1,
            output_shape[0],
        )  # batch size is always 1 for this implementation

        self.n_neurons_current = None
        self.input_shape = None
        self.n_neurons_post = output_shape[0]
        self.batch_size = 0

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

        # This Due a instability in the training process, numeric overflow
        if self.activation_fn == ACTIVATION_FN.RELU:
            scale = np.sqrt(2.0 / n_in)  # He
        elif self.activation_fn in {ACTIVATION_FN.SIGMOID, ACTIVATION_FN.TANH}:
            scale = np.sqrt(1.0 / n_in)  # Xavier
        else:
            scale = 0.01

        self.parameters = Parameters_dense(
            weights=np.random.randn(n_in, n_out) * scale,
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

        A_current = Z_prev @ W + b  # shape (batch_size, n_neurons_post)

        activation = self.activation_fn
        Z_current = None

        if activation == ACTIVATION_FN.RELU:
            Z_current = np.maximum(0, A_current)
        elif activation == ACTIVATION_FN.SIGMOID:
            Z_current = 1 / (1 + np.exp(-A_current))
        elif activation == ACTIVATION_FN.TANH:
            Z_current = np.tanh(A_current)
        elif activation == ACTIVATION_FN.SOFTMAX:
            exp_values = np.exp(A_current - np.max(A_current, axis=1, keepdims=True))
            Z_current = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        return Z_current

    def forward(self, data_in: np.ndarray) -> np.ndarray:
        self.batch_size = data_in.shape[0]

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

        if g == ACTIVATION_FN.SIGMOID:
            s = 1 / (1 + np.exp(-Z_current))
            dg = s * (1 - s)
        elif g == ACTIVATION_FN.RELU:
            dg = np.where(Z_current > 0, 1, 0)
        elif g == ACTIVATION_FN.TANH:
            dg = 1 - np.tanh(Z_current) ** 2
        elif g == ACTIVATION_FN.SOFTMAX:
            dg = 1
        else:
            raise ValueError("Función de activación desconocida: " + g)

        dA = dZ_current * dg  # regla de la cadena dz = dl/dZ * dg/dA

        dW = Z_prev.T @ dA / m  # (n_in, m) @ (m, n_out) = (n_in, n_out)
        db = np.sum(dA, axis=0, keepdims=True) / m  # shape: (m, n_out)
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

        if gradient_in.ndim != 2 or gradient_in.shape[0] != self.batch_size:
            raise ValueError(
                f"Expected shape ({self.batch_size}, n), got {gradient_in.shape}"
            )

        dZ_prev, gradients = self.backward_step(gradient_in)
        self.gradients = gradients

        self.already_forwarded = False
        self.already_backwarded = True

        return dZ_prev

    def update_parameters(self, optimizerConfig: OptimizerConfig | MomentumConfig):
        if not self.is_initialized:
            raise ValueError("Parameters must be initialized before updating.")

        if not self.already_backwarded:
            raise ValueError("Backward pass must be called before updating parameters.")

        if isinstance(optimizerConfig, OptimizerConfig):
            self.parameters.weights -= (
                optimizerConfig.learning_rate * self.gradients.g_weights
            )
            self.parameters.biases -= (
                optimizerConfig.learning_rate * self.gradients.g_biases
            )
        elif isinstance(optimizerConfig, MomentumConfig):
            if self.vW is None:
                self.vW = np.zeros_like(self.parameters.W)
            if self.vb is None:
                self.vb = np.zeros_like(self.parameters.b)

            self.vW = (
                optimizerConfig.momentum * self.vW
                - optimizerConfig.learning_rate * self.gradient.dW
            )
            self.vb = (
                optimizerConfig.momentum * self.vb
                - optimizerConfig.learning_rate * self.gradient.db
            )
            self.parameters.W += self.vW
            self.parameters.b += self.vb
        else:
            raise ValueError(
                f"Unsupported optimizerConfig type: {type(optimizerConfig)}. Expected OptimizerConfig or MomentumConfig."
            )

        self.already_backwarded = False
        return self.parameters

    def get_activation_function(self) -> ACTIVATION_FN:
        return self.activation_fn

    def get_parameters(self):
        return self.parameters

    def get_gradients(self):
        return self.gradients
