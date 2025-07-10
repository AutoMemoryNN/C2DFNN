from dataclasses import dataclass
from enum import Enum

import numpy as np

from Layer import Layer, Layers_type, Activation_fn
from LayerDense_v2 import LayerDense
from LayerConv import (
    LayerConv,
    LayerFlatten,
    LayerPooling,
    Pooling_fn,
    Specification_conv,
    Specification_pooling,
)


class loss_fn(Enum):
    CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
    MEAN_SQUARED_ERROR = "mse"
    BINARY_CROSSENTROPY = "binary_crossentropy"


@dataclass
class LayersConfig:
    layersConfig: list[Layer]


class Network:
    def __init__(self, layers: LayersConfig) -> None:
        self.layers = layers.layersConfig
        self.init_layers()
        self.parameters = {}  # TODO: Necessary?
        self.activations = {}

    def init_layers(self):
        self.parameters = {}
        self.activations = {}

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if not isinstance(layer, Layer):
                raise TypeError(
                    f"Layer at index {i} is not an instance of Layer class."
                )

            if layer.name is None:
                if layer.get_layer_type() == Layers_type.CONVOLUTIONAL:
                    layer.name = f"conv_layer_{i}"
                elif layer.get_layer_type() == Layers_type.FLATTEN:
                    layer.name = f"flatter_layer_{i}"
                elif layer.get_layer_type() == Layers_type.DENSE:
                    layer.name = f"dense_layer_{i}"

            previous_layer = self.layers[i - 1]

            if i == 0:
                if layer.get_input_shape() is None or layer.output_shape is None:
                    raise ValueError(
                        f"Input and output shapes must be defined for the first layer: {layer.name}"
                    )
            else:
                layer.set_input_shape(previous_layer.get_output_shape())

            layer.initialize_parameters()

            self.parameters[layer.name] = {}
            self.activations[layer.name] = {}

    def cost(self, Yp: np.ndarray, Y: np.ndarray, costfunction: loss_fn) -> np.ndarray:
        output_layer = self.layers[-1]
        if (
            costfunction == loss_fn.CATEGORICAL_CROSSENTROPY
            and output_layer.get_activation_function() == Activation_fn.SOFTMAX
        ):
            return -np.sum(Y * np.log(Yp + 1e-15), axis=1)
        elif (
            costfunction == loss_fn.MEAN_SQUARED_ERROR
            and output_layer.get_activation_function() == Activation_fn.RELU
        ):
            return np.mean(np.square(Y - Yp), axis=1)
        elif (
            costfunction == loss_fn.BINARY_CROSSENTROPY
            and output_layer.get_activation_function() == Activation_fn.SIGMOID
        ):
            return -np.mean(
                Y * np.log(Yp + 1e-15) + (1 - Y) * np.log(1 - Yp + 1e-15), axis=1
            )
        else:
            raise ValueError(
                f"Unsupported cost function or cost functions {costfunction} is not compatible with the output layer's activation function {output_layer.get_activation_function()}."
            )

    def d_cost(
        self, Yp: np.ndarray, Y: np.ndarray, costfunction: loss_fn
    ) -> np.ndarray:
        if costfunction == loss_fn.CATEGORICAL_CROSSENTROPY:
            return -Y / (Yp + 1e-15)
        elif costfunction == loss_fn.MEAN_SQUARED_ERROR:
            return 2 * (Yp - Y) / Y.shape[0]
        elif costfunction == loss_fn.BINARY_CROSSENTROPY:
            return -Y / (Yp + 1e-15) + (1 - Y) / (1 - Yp + 1e-15)
        else:
            raise ValueError(
                f"Unsupported cost function: {costfunction}. Supported functions are: {list(loss_fn)}."
            )

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cost_function: loss_fn,
        learning_rate: float = 0.01,
        epochs: int = 50,
        print_cost=False,
        show_graph=False,
    ):  # user cannot choice the loss function
        costs = []
        for epoch in range(epochs):
            # Forward pass
            A = X
            for layer in self.layers:
                A = layer.forward(A)
                self.activations[layer.name] = A
            # Compute cost
            cost = self.cost(A, Y, cost_function)
            costs.append(cost)

            # Backward pass
            dA = self.d_cost(A, Y, cost_function)
            for layer in reversed(self.layers):
                dA = layer.backward(dA)
                self.parameters[layer.name] = layer.get_parameters()

            # Update parameters
            for layer in self.layers:
                layer.update_parameters(learning_rate)

    def one_hot_encode(self, y, num_classes):
        pass


if __name__ == "__main__":

    layers = [
        LayerConv(
            input_shape=(1, 28, 28, 1),
            specification=Specification_conv(
                c_filter=3,
                c_channels=1,
                c_filters=32,
                c_stride=1,
                c_pad=0,
                activation=Activation_fn.RELU,
            ),
        ),
        LayerPooling(
            specification=Specification_pooling(
                p_filter=2, p_stride=2, p_function=Pooling_fn.MAX
            ),
            name="pooling_layer_interesting",
        ),
        LayerConv(
            input_shape=(1, 26, 26, 32),
            specification=Specification_conv(
                c_filter=3,
                c_channels=32,
                c_filters=64,
                c_stride=1,
                c_pad=0,
                activation=Activation_fn.RELU,
            ),
        ),
        LayerPooling(
            specification=Specification_pooling(
                p_filter=2, p_stride=2, p_function=Pooling_fn.MAX
            ),
        ),
        LayerFlatten(),
        LayerDense(
            (128,),
            Activation_fn.RELU,
        ),
        LayerDense(
            (64,),
            Activation_fn.RELU,
        ),
        LayerDense(
            (10,),
            Activation_fn.SOFTMAX,
            name="output_layer",
        ),
    ]

    # You can now use layers for further processing
    # For example: network = Network(LayersConfig(layers))
