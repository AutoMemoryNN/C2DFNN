from dataclasses import dataclass
from enum import Enum
from typing import Literal, Tuple
import numpy as np

from Layer import Layer, Layers_type, Activation_fn


class Pooling_fn(Enum):
    MAX = "max"
    AVERAGE = "average"


@dataclass
class Specification_conv:
    c_filter: int
    c_channels: int
    c_filters: int
    c_stride: int
    c_pad: int
    activation: Activation_fn


@dataclass
class Specification_pooling:
    p_filter: int
    p_stride: int
    p_function: Pooling_fn


@dataclass
class Cache_conv:
    data_in: np.ndarray
    data_conv: np.ndarray
    data_act: np.ndarray


@dataclass
class Cache_pooling:
    data_act: np.ndarray
    data_pool: np.ndarray


@dataclass
class Parameters:
    W: np.ndarray  # (f, f, c, n_f)
    b: np.ndarray  # (1, 1, 1, n_f)


@dataclass
class Gradient:
    dW: np.ndarray
    db: np.ndarray


class LayerFlatten(Layer):
    """
    Flatten Layer class
    Converts 2D Convolutional Layer output to a Dense Layer input
    """

    def __init__(self, name: str | None = None):
        super().__init__(layer_type=Layers_type.FLATTEN, name=name)
        self.previous_shape = None

    def forward(self, data_in: np.ndarray) -> np.ndarray:
        self.previous_shape = data_in.shape
        return self._to_dense(data_in)

    def backward(self, gradient_in: np.ndarray) -> np.ndarray:
        if self.previous_shape is None:
            raise ValueError("forward must be called before backward")

        return gradient_in.reshape(self.previous_shape)

    def _to_dense(self, data_in):
        """
        Convert 2D Convolutional Layer output to a Dense Layer input
        Arguments:
        data_in – Input data with shape (batch_size, height, width, channels)
        Returns:
        data_out – Flattened output with shape (batch_size, height * width * channels)
        """
        if not isinstance(data_in, np.ndarray):
            raise TypeError("data_in must be a numpy array")

        if data_in.ndim != 4:
            raise ValueError(
                f"Expected data_in to have 4 dimensions (batch, height, width, channels), got {data_in.ndim}"
            )

        batch_size, height, width, channels = data_in.shape

        # Flatten all dimensions except the batch dimension
        # Shape: (batch_size, height * width * channels)
        data_out = data_in.reshape(batch_size, height * width * channels)

        return data_out

    def set_input_shape(self, input_shape: tuple[int, int, int, int] | None):
        if input_shape is None or len(input_shape) != 4:
            raise ValueError(
                f"Input shape must be a tuple of 4 integers: (batch_size, height, width, channels), got {input_shape}"
            )
        self.input_shape = input_shape

        # Output shape is determined by the flatten operation
        self.output_shape = self._to_dense(np.zeros(input_shape)).shape

    def initialize_parameters(self):
        """
        Initialize parameters for the flatten layer.
        Flatten layers typically do not have learnable parameters, so this method can be empty.
        """
        pass

    def get_activation_function(self) -> Activation_fn | None:
        return None

    def get_parameters(self):
        return Parameters(W=np.array([]), b=np.array([]))

    def get_gradients(self):
        return Gradient(dW=np.array([]), db=np.array([]))

    def update_parameters(self, learning_rate: float):
        pass


# TODO: continue form Gradient
class LayerConv(Layer):
    """
    Convolutional Layer class
    """

    def __init__(
        self,
        specification: Specification_conv,
        input_shape: (
            Tuple[int, int, int, int] | None
        ) = None,  # (batch_size, height, width, channels),
        name: str | None = None,
    ):
        super().__init__(
            layer_type=Layers_type.CONVOLUTIONAL,
            input_shape=input_shape,
            name=name,
        )

        self.cache = Cache_conv(
            data_in=np.ndarray([]),
            data_conv=np.ndarray([]),
            data_act=np.ndarray([]),
        )
        self.specification = specification

        if input_shape is not None:
            self.generate_output_shape()

        self.parameters = Parameters(
            W=np.ndarray([]),
            b=np.ndarray([]),
        )

        self.gradient = Gradient(
            dW=np.ndarray([]),
            db=np.ndarray([]),
        )

    def set_input_shape(self, input_shape: tuple[int, int, int, int] | None):
        if input_shape is None or len(input_shape) != 4:
            raise ValueError(
                f"Input shape must be a tuple of 4 integers: (batch_size, height, width, channels), got {input_shape}"
            )
        self.input_shape = input_shape

        self.generate_output_shape()

    def generate_output_shape(self):
        batch_size = self.input_shape[0] if self.input_shape[0] != -1 else 1

        self.output_shape = (
            batch_size,
            int(
                (
                    self.input_shape[1]
                    - self.specification.c_filter
                    + 2 * self.specification.c_pad
                )
                / self.specification.c_stride
                + 1
            ),
            int(
                (
                    self.input_shape[2]
                    - self.specification.c_filter
                    + 2 * self.specification.c_pad
                )
                / self.specification.c_stride
                + 1
            ),
            self.specification.c_filters,
        )

    def backward(self, gradient_in: np.ndarray) -> np.ndarray:
        data_out, gradient = self._backward_convolutional(gradient_in)
        return data_out

    def initialize_parameters(self):
        filter_size = self.specification.c_filter
        n_channels = self.specification.c_channels
        n_filters = self.specification.c_filters
        activation = self.specification.activation

        W_shape = (
            filter_size,
            filter_size,
            n_channels,
            n_filters,
        )

        fan_in = filter_size * filter_size * n_channels

        # This Due a instability in the training process, numeric overflow
        if activation == Activation_fn.RELU:
            scale = np.sqrt(2.0 / fan_in)  # He
        elif activation in {Activation_fn.SIGMOID, Activation_fn.TANH}:
            scale = np.sqrt(1.0 / fan_in)  # Xavier
        else:
            scale = 0.01

        self.parameters.W = np.random.randn(*W_shape) * scale
        self.parameters.b = np.zeros((1, 1, 1, n_filters))

        self.gradient.dW = np.zeros_like(self.parameters.W, dtype=np.float64)
        self.gradient.db = np.zeros_like(self.parameters.b, dtype=np.float64)

    def _add_pad(self, data_in: np.ndarray, pad: int) -> np.ndarray:
        """
        Pad all examples of data_in
        Arguments:
        data_in –
        pad -
        Returns:
        data_out
        """
        data_out = np.pad(
            array=data_in,
            pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return data_out

    def _one_convolution(self, X: np.ndarray, W: np.ndarray, b: float) -> float:
        """
        Calculate one convolution value
        Arguments:
        X –- Array (c_filter, c_filter, c_channels) (slice of data)
        W –- Array (c_filter, c_filter, c_channels)
        b – float
        Returns:
        v -- value
        """
        if X.shape != W.shape:
            raise ValueError(
                f"X and W must have the same shape, got X: {X.shape}, W: {W.shape}"
            )

        v = np.sum(X * W) + b
        return v

    def _forward_convolution_step(
        self,
        data_in: np.ndarray,
        specification: Specification_conv,
        parameters: Parameters,
    ) -> np.ndarray:
        """
        Do a forward convolution step
        Arguments:
        data_in –-
        specification –-
        parameters --
        Returns:
        data_out
        """

        padding = specification.c_pad
        stride = specification.c_stride
        filter_size = specification.c_filter
        n_filters = specification.c_filters

        Xp = self._add_pad(data_in, padding)

        _, h_in, w_in, n_channels = Xp.shape

        h_out = int((h_in - filter_size) // stride + 1)
        w_out = int((w_in - filter_size) // stride + 1)

        data_out = np.zeros((1, h_out, w_out, n_filters))

        for h in range(h_out):
            for w in range(w_out):
                for f in range(n_filters):
                    h_start = h * stride
                    h_end = h_start + filter_size
                    w_start = w * stride
                    w_end = w_start + filter_size

                    X_slice = Xp[0, h_start:h_end, w_start:w_end, :]

                    X_slice_adjusted = X_slice[:, :, 0 : specification.c_channels]

                    W_slice = parameters.W[:, :, :, f]
                    b_slice = parameters.b[0, 0, 0, f]

                    data_out[0, h, w, f] = self._one_convolution(
                        X_slice_adjusted, W_slice, b_slice
                    )

        return data_out

    def _forward_activation_step(
        self, data_in: np.ndarray, specification: Specification_conv
    ) -> np.ndarray:
        """
        Do a forward activation step
        Arguments:
        data_in –-
        specification –-
        Returns:
        data_out --
        """

        fnc = specification.activation

        if fnc == Activation_fn.RELU:
            return np.maximum(0, data_in)
        elif fnc == Activation_fn.SIGMOID:
            return 1 / (1 + np.exp(-data_in))
        elif fnc == Activation_fn.TANH:
            return np.tanh(data_in)
        else:
            raise ValueError(f"Unsupported activation function: {fnc}")

    def forward(
        self,
        data_in: np.ndarray,
    ) -> np.ndarray:
        """
        Do forward convolutional propagation
        Arguments:
        data_in –-
        specification –-
        parameters --
        Returns:
        data_out --
        cache --
        """
        if not isinstance(data_in, np.ndarray):
            raise TypeError("data_in must be a numpy array")

        if data_in.ndim != 4:
            raise ValueError(
                f"Expected data_in to have 4 dimensions, got {data_in.ndim}"
            )

        # Check if parameters shape matches specification
        expected_w_shape = (
            self.specification.c_filter,
            self.specification.c_filter,
            self.specification.c_channels,
            self.specification.c_filters,
        )
        if self.parameters.W.shape != expected_w_shape:
            raise ValueError(
                f"Parameters W shape {self.parameters.W.shape} doesn't match expected {expected_w_shape}"
            )

        data_conv = self._forward_convolution_step(
            data_in, self.specification, self.parameters
        )

        data_act = self._forward_activation_step(data_conv, self.specification)

        self.cache = Cache_conv(
            data_in=data_in,
            data_conv=data_conv,
            data_act=data_act,
        )

        return data_act

    def _backward_activation_step(self, gradient_in):
        """
        Do a backward activation step
        Arguments:
        gradient_in –-
        specification –-
        cache --
        Returns:
        cost_gradient_out –-
        parameters_gradient --
        """
        activation = self.specification.activation
        if activation == Activation_fn.RELU:
            dZ = np.where(self.cache.data_conv > 0, gradient_in, 0)
        elif activation == Activation_fn.SIGMOID:
            sig = self.cache.data_act
            dZ = sig * (1 - sig) * gradient_in
        elif activation == Activation_fn.TANH:
            dZ = (1 - np.square(self.cache.data_act)) * gradient_in
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Activation has no parameters
        dW = np.array([])
        db = np.array([])

        return dZ, Gradient(dW, db)

    def _backward_convolution_step(self, gradient_in):
        """
        Do a backward convolution step
        Arguments:
        gradient_in –-
        Returns:
        cost_gradient_out –-
        parameters_gradient --
        """
        stride = self.specification.c_stride
        filter_size = self.specification.c_filter
        n_filters = self.specification.c_filters
        pad = self.specification.c_pad

        # Get dimensions
        m, h_out, w_out, n_f = gradient_in.shape

        # Add padding to input data for gradient computation
        data_in_padded = self._add_pad(self.cache.data_in, pad)

        # Initialize gradients with correct shapes and ensure float dtype
        dW = np.zeros_like(self.parameters.W, dtype=np.float64)
        db = np.zeros_like(self.parameters.b, dtype=np.float64)
        dA_prev = np.zeros_like(data_in_padded, dtype=np.float64)

        # Ensure gradient_in is float type
        gradient_in = gradient_in.astype(np.float64)

        # Compute gradients
        for h in range(h_out):
            for w in range(w_out):
                for f in range(n_f):
                    # Define slice boundaries
                    h_start = h * stride
                    h_end = h_start + filter_size
                    w_start = w * stride
                    w_end = w_start + filter_size

                    # Extract slice from padded input and ensure float type
                    a_slice = data_in_padded[0, h_start:h_end, w_start:w_end, :].astype(
                        np.float64
                    )

                    # FIX: Make sure we only use the channels that match the filter
                    # The filter expects c_channels, so slice accordingly
                    n_channels = self.specification.c_channels
                    a_slice_corrected = a_slice[:, :, :n_channels]

                    # Update gradients
                    dW[:, :, :, f] += a_slice_corrected * gradient_in[0, h, w, f]
                    db[0, 0, 0, f] += gradient_in[0, h, w, f]

                    # Update gradient w.r.t. input
                    # Only propagate to the channels that were actually used
                    dA_prev[0, h_start:h_end, w_start:w_end, :n_channels] += (
                        self.parameters.W[:, :, :, f].astype(np.float64)
                        * gradient_in[0, h, w, f]
                    )

        # Remove padding from input gradients if padding was applied
        if pad > 0:
            dA_prev = dA_prev[:, pad:-pad, pad:-pad, :]

        return dA_prev, Gradient(dW, db)

    def _backward_convolutional(self, gradient_in):
        """
        Do backward convolutional layer propagation
        Arguments:
        gradient_in –-
        specification –-
        cache --
        parameters --
        Returns:
        cost_gradient_out –-
        parameters_gradient --
        """
        dA_act, gradient_act = self._backward_activation_step(gradient_in)

        dA_conv, grad_conv = self._backward_convolution_step(dA_act)

        # only convolutional parameters went
        total_gradient = Gradient(dW=grad_conv.dW, db=grad_conv.db)

        return dA_conv, total_gradient

    def update_parameters(self, learning_rate: float):
        """
        Update parameters using gradient descent
        Arguments:
        parameters –- current parameters
        gradient –- computed gradients
        learning_rate –- learning rate for the update
        Returns:
        updated_parameters -- updated parameters after applying the gradients
        """
        self.parameters.W -= learning_rate * self.gradient.dW
        self.parameters.b -= learning_rate * self.gradient.db

    def get_activation_function(self) -> Activation_fn | None:
        return self.specification.activation

    def get_parameters(self):
        return self.parameters

    def get_gradients(self):
        raise NotImplementedError


class LayerPooling(Layer):
    """
    Pooling Layer class
    """

    def __init__(
        self,
        specification: Specification_pooling,
        name: str | None = None,
    ):
        super().__init__(
            layer_type=Layers_type.POOLING,
            name=name,
        )
        self.cache = Cache_pooling(
            data_act=np.ndarray([]),
            data_pool=np.ndarray([]),
        )
        self.specification = specification

    def set_input_shape(self, input_shape: tuple[int, int, int, int] | None):
        if input_shape is None or len(input_shape) != 4:
            raise ValueError(
                f"Input shape must be a tuple of 4 integers: (batch_size, height, width, channels), got {input_shape}"
            )
        self.input_shape = input_shape
        self.generate_output_shape()

    def generate_output_shape(self):
        if self.input_shape is None:
            raise ValueError("Input shape must be set before generating output shape")

        h_out = (
            self.input_shape[1] - self.specification.p_filter
        ) // self.specification.p_stride + 1
        w_out = (
            self.input_shape[2] - self.specification.p_filter
        ) // self.specification.p_stride + 1
        self.output_shape = (1, h_out, w_out, self.input_shape[3])

    def forward(self, data_in: np.ndarray) -> np.ndarray:
        data_out = self._forward_pooling_step(data_in)

        self.cache = Cache_pooling(
            data_act=data_in,
            data_pool=data_out,
        )

        return data_out

    def backward(self, gradient_in: np.ndarray) -> np.ndarray:
        gradient_out, gradient = self._backward_pooling_step(gradient_in)
        return gradient_out

    def initialize_parameters(self):
        """
        Initialize parameters for the pooling layer.
        Pooling layers typically do not have learnable parameters, so this method can be empty.
        """
        pass

    def _forward_pooling_step(self, data_in: np.ndarray) -> np.ndarray:
        """
        Do a forward pooling step
        Arguments:
        data_in –-
        specification –-
        Returns:
        data_out --
        """
        fnc = self.specification.p_function
        stride = self.specification.p_stride
        filter_size = self.specification.p_filter

        n_filters = data_in.shape[-1]
        h_in, w_in = data_in.shape[1:3]

        h_out = (h_in - filter_size) // stride + 1
        w_out = (w_in - filter_size) // stride + 1

        data_out = np.zeros((1, h_out, w_out, n_filters))

        for f in range(n_filters):
            for h in range(h_out):
                for w in range(w_out):
                    h_start = h * stride
                    h_end = h_start + filter_size
                    w_start = w * stride
                    w_end = w_start + filter_size

                    X_slice = data_in[
                        0, h_start:h_end, w_start:w_end, f
                    ]  # assuming batch size of 1

                    if fnc == Pooling_fn.MAX:
                        data_out[0, h, w, f] = np.max(X_slice)
                    elif fnc == Pooling_fn.AVERAGE:
                        data_out[0, h, w, f] = np.mean(X_slice)
                    else:
                        raise ValueError(f"Unsupported pooling function: {fnc}")

        return data_out

    def _backward_pooling_step(
        self,
        gradient_in: np.ndarray,
    ) -> Tuple[np.ndarray, Gradient]:
        """
        Do a backward pooling step
        Arguments:
        gradient_in –-
        specification –-
        cache --
        Returns:
        cost_gradient_out –-
        parameters_gradient --
        """

        data_post_act = self.cache.data_act

        dZ = np.zeros_like(data_post_act)

        # Get dimensions from gradient_in (pooled output), not from dZ
        _, h_grad, w_grad, n_filters = gradient_in.shape

        pool_size = self.specification.p_filter
        stride = self.specification.p_stride

        for p in range(h_grad):
            for q in range(w_grad):
                for current_filter in range(n_filters):
                    h_start = p * stride
                    h_end = h_start + pool_size
                    w_start = q * stride
                    w_end = w_start + pool_size

                    # Ensure we don't go out of bounds
                    h_end = min(h_end, data_post_act.shape[1])
                    w_end = min(w_end, data_post_act.shape[2])

                    # Window
                    window = data_post_act[
                        0, h_start:h_end, w_start:w_end, current_filter
                    ]

                    if window.size == 0:  # Skip empty windows
                        continue

                    if self.specification.p_function == Pooling_fn.MAX:
                        idx = np.unravel_index(np.argmax(window), window.shape)
                        i_max, j_max = idx

                        dZ[
                            0, h_start + i_max, w_start + j_max, current_filter
                        ] += gradient_in[0, p, q, current_filter]

                    elif self.specification.p_function == Pooling_fn.AVERAGE:
                        avg_grad = gradient_in[0, p, q, current_filter] / window.size
                        dZ[0, h_start:h_end, w_start:w_end, current_filter] += avg_grad

                    else:
                        raise ValueError(
                            f"Unsupported pooling function: {self.specification.p_function}"
                        )

        # Pooling has no parameters
        dW = np.array([])
        db = np.array([])

        return dZ, Gradient(dW, db)

    def get_activation_function(self) -> Activation_fn | None:
        return None

    def get_parameters(self):
        return Parameters(W=np.array([]), b=np.array([]))

    def get_gradients(self):
        return Gradient(dW=np.array([]), db=np.array([]))

    def update_parameters(self, learning_rate: float):
        pass
