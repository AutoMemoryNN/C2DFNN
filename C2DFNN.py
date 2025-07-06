from dataclasses import dataclass
from enum import Enum

from Layer import Layer, Layers_type, Activation_fn
from LayerDense import LayerDense
from LayerConv import (
    LayerConv,
    LayerPooling,
    Pooling_fn,
    Specification_conv,
    Specification_pooling,
)


class loss(Enum):
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
        self.parameters = {}
        self.activations = {}
        self.initialize_parameters(self.layers)

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
                elif layer.get_layer_type() == Layers_type.FLATTER:
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

    def initialize_parameters(self, layers):
        pass

    def predict(self, X, parameters, activations):
        pass

    def accuracy(self, X, Y, parameters, activations):
        pass

    def mse(self, X, Y, parameters, activations):
        pass

    def load_y_data(self, file_path):
        pass

    def cost(self, Yp, Y, costfunction):
        pass

    def update_parameters(self, parameters, grads, learning_rate):
        pass

    def train(
        self,
        X,
        Y,
        hidden_layers,
        learning_rate,
        num_iterations,
        costfunction,
        activations,
        print_cost=False,
    ):
        pass

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
    ]

    # You can now use layers for further processing
    # For example: network = Network(LayersConfig(layers))
