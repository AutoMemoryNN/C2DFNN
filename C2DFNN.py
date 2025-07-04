from dataclasses import dataclass
from enum import Enum

from Layer import Layer


class layers(Enum):
    CONVOLUTIONAL = "convolutional"
    FLATTER = "flatter"
    DENSE = "dense"


class activation(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"


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
                if layer.get_layer_type() == layers.CONVOLUTIONAL:
                    layer.name = f"conv_layer_{i}"
                elif layer.get_layer_type() == layers.FLATTER:
                    layer.name = f"flatter_layer_{i}"
                elif layer.get_layer_type() == layers.DENSE:
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

    pass
