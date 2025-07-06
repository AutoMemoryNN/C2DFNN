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
class Cache:
    data_in: np.ndarray
    data_conv: np.ndarray
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


def add_pad(data_in: np.ndarray, pad: int) -> np.ndarray:
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


def one_convolution(X: np.ndarray, W: np.ndarray, b: int) -> int:
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


def forward_convolution_step(
    data_in: np.ndarray, specification: Specification_conv, parameters: Parameters
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

    Xp = add_pad(data_in, padding)

    _, h_in, w_in, n_channels = Xp.shape
    print(f"Input shape: {Xp.shape}")

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

                data_out[0, h, w, f] = one_convolution(
                    X_slice_adjusted, W_slice, b_slice
                )

    return data_out


def forward_activation_step(
    data_in: np.ndarray, specification: Specification_conv
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


def forward_pooling_step(
    data_in: np.ndarray, specification: Specification_pooling
) -> np.ndarray:
    """
    Do a forward pooling step
    Arguments:
    data_in –-
    specification –-
    Returns:
    data_out --
    """
    fnc = specification.p_function
    stride = specification.p_stride
    filter_size = specification.p_filter

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


def forward_convolutional(
    data_in: np.ndarray,
    specification_conv: Specification_conv,
    specification_pooling: Specification_pooling,
    parameters: Parameters,
) -> Tuple[np.ndarray, Cache]:
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
        raise ValueError(f"Expected data_in to have 4 dimensions, got {data_in.ndim}")

    # Check if parameters shape matches specification
    expected_w_shape = (
        specification_conv.c_filter,
        specification_conv.c_filter,
        specification_conv.c_channels,
        specification_conv.c_filters,
    )
    if parameters.W.shape != expected_w_shape:
        raise ValueError(
            f"Parameters W shape {parameters.W.shape} doesn't match expected {expected_w_shape}"
        )

    data_conv = forward_convolution_step(data_in, specification_conv, parameters)

    data_act = forward_activation_step(data_conv, specification_conv)

    data_out = forward_pooling_step(data_act, specification_pooling)

    return (
        data_out,
        Cache(
            data_in=data_in, data_conv=data_conv, data_act=data_act, data_pool=data_out
        ),
    )


# ____________________________________ backprop__________________________________________


def backward__pooling_step(
    gradient_in: np.ndarray, specification: Specification_pooling, cache: Cache
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

    dZ = np.zeros_like(cache.data_act)

    # Get dimensions from gradient_in (pooled output), not from dZ
    _, h_grad, w_grad, n_filters = gradient_in.shape

    pool_size = specification.p_filter
    stride = specification.p_stride

    for p in range(h_grad):
        for q in range(w_grad):
            for current_filter in range(n_filters):
                h_start = p * stride
                h_end = h_start + pool_size
                w_start = q * stride
                w_end = w_start + pool_size

                # Ensure we don't go out of bounds
                h_end = min(h_end, cache.data_act.shape[1])
                w_end = min(w_end, cache.data_act.shape[2])

                # Window
                window = cache.data_act[0, h_start:h_end, w_start:w_end, current_filter]

                if window.size == 0:  # Skip empty windows
                    continue

                if specification.p_function == Pooling_fn.MAX:
                    idx = np.unravel_index(np.argmax(window), window.shape)
                    i_max, j_max = idx

                    dZ[
                        0, h_start + i_max, w_start + j_max, current_filter
                    ] += gradient_in[0, p, q, current_filter]

                elif specification.p_function == Pooling_fn.AVERAGE:
                    avg_grad = gradient_in[0, p, q, current_filter] / window.size
                    dZ[0, h_start:h_end, w_start:w_end, current_filter] += avg_grad

                else:
                    raise ValueError(
                        f"Unsupported pooling function: {specification.p_function}"
                    )

    # Pooling has no parameters
    dW = np.array([])
    db = np.array([])

    return dZ, Gradient(dW, db)


def backward_activation_step(gradient_in, specification, cache):
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
    if specification.activation == Activation_fn.RELU:
        dZ = np.where(cache.data_conv > 0, gradient_in, 0)
    elif specification.activation == Activation_fn.SIGMOID:
        sig = cache.data_act
        dZ = sig * (1 - sig) * gradient_in
    elif specification.activation == Activation_fn.TANH:
        dZ = (1 - np.square(cache.data_act)) * gradient_in
    else:
        raise ValueError(f"Unsupported activation function: {specification.activation}")

    # Activation has no parameters
    dW = np.array([])
    db = np.array([])

    return dZ, Gradient(dW, db)


def backward_convolution_step(gradient_in, specification, cache, parameters):
    """
    Do a backward convolution step
    Arguments:
    gradient_in –-
    specification –-
    cache --
    parameters --
    Returns:
    cost_gradient_out –-
    parameters_gradient --
    """
    stride = specification.c_stride
    filter_size = specification.c_filter
    n_filters = specification.c_filters
    pad = specification.c_pad

    # Get dimensions
    m, h_out, w_out, n_f = gradient_in.shape

    # Add padding to input data for gradient computation
    data_in_padded = add_pad(cache.data_in, pad)

    # Initialize gradients with correct shapes and ensure float dtype
    dW = np.zeros_like(parameters.W, dtype=np.float64)
    db = np.zeros_like(parameters.b, dtype=np.float64)
    dA_prev = np.zeros_like(data_in_padded, dtype=np.float64)

    # Ensure gradient_in is float type
    gradient_in = gradient_in.astype(np.float64)

    # Compute gradients
    for h in range(h_out):
        for w in range(w_out):
            for f in range(n_filters):
                # Define slice boundaries
                h_start = h * stride
                h_end = h_start + filter_size
                w_start = w * stride
                w_end = w_start + filter_size

                # Extract slice from padded input and ensure float type
                a_slice = data_in_padded[0, h_start:h_end, w_start:w_end, :].astype(
                    np.float64
                )

                # Update gradients
                dW[:, :, :, f] += a_slice * gradient_in[0, h, w, f]
                db[0, 0, 0, f] += gradient_in[0, h, w, f]

                # Update gradient w.r.t. input
                dA_prev[0, h_start:h_end, w_start:w_end, :] += (
                    parameters.W[:, :, :, f].astype(np.float64)
                    * gradient_in[0, h, w, f]
                )

    # Remove padding from input gradients if padding was applied
    if pad > 0:
        dA_prev = dA_prev[:, pad:-pad, pad:-pad, :]

    return dA_prev, Gradient(dW, db)


def backward_convolutional(gradient_in, specification, cache, parameters):
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
    # Step 1: Backward through pooling
    dA_pool, grad_pool = backward__pooling_step(gradient_in, specification, cache)

    # Step 2: Backward through activation
    dA_act, grad_act = backward_activation_step(dA_pool, specification, cache)

    # Step 3: Backward through convolution
    dA_conv, grad_conv = backward_convolution_step(
        dA_act, specification, cache, parameters
    )

    # Combine gradients (pooling and activation have no parameters, only convolution does)
    total_gradient = Gradient(dW=grad_conv.dW, db=grad_conv.db)

    return dA_conv, total_gradient


def to_dense(data_in):
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


class LayerConv(Layer):
    """
    Convolutional Layer class
    """

    def __init__(
        self,
        input_shape: Tuple[
            int, int, int, int
        ],  # (batch_size, height, width, channels),
        specification: Specification_conv,
        name: str | None = None,
    ):
        super().__init__(
            layer_type=Layers_type.CONVOLUTIONAL,
            input_shape=input_shape,
            name=name,
        )
        self.specification = specification

    def forward(self, data_in: np.ndarray):
        pass

    def backward(self, gradient_in: np.ndarray, cache: Cache):
        pass

    def initialize_parameters(self):
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
            layer_type=Layers_type.CONVOLUTIONAL,
            name=name,
        )
        self.specification = specification

    def forward(self, data_in: np.ndarray):
        pass

    def backward(self, gradient_in: np.ndarray, cache: Cache):
        pass

    def initialize_parameters(self):
        raise NotImplementedError
