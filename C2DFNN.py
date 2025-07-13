import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from tensorflow.keras.datasets import mnist  # type: ignore
from dataclasses import dataclass
from enum import Enum
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
                    layer.name = f"conv_layer_{i} "
                elif layer.get_layer_type() == Layers_type.DENSE:
                    layer.name = f"dense_layer_{i}"
                elif layer.get_layer_type() == Layers_type.FLATTEN:
                    layer.name = f"flat_layer_{i} "
                elif layer.get_layer_type() == Layers_type.POOLING:
                    layer.name = f"pool_layer_{i} "

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        Yp = self.predict(X)
        output_activation = self.layers[-1].get_activation_function()

        if output_activation == Activation_fn.SOFTMAX:
            Yp_labels = np.argmax(Yp, axis=1)
            Y_true = np.argmax(Y, axis=1)
        elif output_activation == Activation_fn.SIGMOID:
            Yp_labels = (Yp > 0.5).astype(int)
            Y_true = Y.astype(int)
        else:
            raise ValueError(
                f"Accuracy not implemented for activation: {output_activation}"
            )

        acc = np.mean(Yp_labels == Y_true)
        return acc

    def cost(self, Yp: np.ndarray, Y: np.ndarray, costfunction: loss_fn) -> np.ndarray:
        output_activation = self.layers[-1].get_activation_function()

        # clipped values to avoid log(0)
        if (
            costfunction == loss_fn.CATEGORICAL_CROSSENTROPY
            and output_activation == Activation_fn.SOFTMAX
        ):
            Yp_clipped = np.clip(Yp, 1e-15, 1 - 1e-15)
            return -np.sum(Y * np.log(Yp_clipped), axis=1)
        elif (
            costfunction == loss_fn.MEAN_SQUARED_ERROR
            and output_activation == Activation_fn.RELU
        ):
            return np.mean(np.square(Y - Yp), axis=1)
        elif (
            costfunction == loss_fn.BINARY_CROSSENTROPY
            and output_activation == Activation_fn.SIGMOID
        ):
            Yp_clipped = np.clip(Yp, 1e-15, 1 - 1e-15)
            return -np.mean(
                Y * np.log(Yp_clipped) + (1 - Y) * np.log(1 - Yp_clipped), axis=1
            )
        else:
            raise ValueError(
                f"Unsupported cost {costfunction} for activation {output_activation}."
            )

    def d_cost(
        self, Yp: np.ndarray, Y: np.ndarray, costfunction: loss_fn
    ) -> np.ndarray:
        if costfunction == loss_fn.CATEGORICAL_CROSSENTROPY:
            return Yp - Y
        elif costfunction == loss_fn.MEAN_SQUARED_ERROR:
            return 2 * (Yp - Y) / Y.shape[0]
        elif costfunction == loss_fn.BINARY_CROSSENTROPY:
            Yp_clipped = np.clip(Yp, 1e-15, 1 - 1e-15)
            return -Y / Yp_clipped + (1 - Y) / (1 - Yp_clipped)
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
        batch_size: int = 16,
        print_cost=False,
        show_graph=False,
    ):
        costs = []

        for epoch in range(epochs):
            epoch_costs = []

            for i in range((X.shape[0] + batch_size - 1) // batch_size):
                x = X[i * batch_size : (i + 1) * batch_size]
                y = Y[i * batch_size : (i + 1) * batch_size]

                # Forward pass
                Z = x
                for layer in self.layers:
                    Z = layer.forward(Z)
                    self.activations[layer.name] = Z

                # Cost
                cost = self.cost(Z, y, cost_function)
                epoch_costs.append(cost)

                # Backward
                dA = self.d_cost(Z, y, cost_function)
                for layer in reversed(self.layers):
                    dA = layer.backward(dA)
                    self.parameters[layer.name] = layer.get_parameters()

                # Update
                for layer in self.layers:
                    layer.update_parameters(learning_rate)

                if i % 10 == 0 and print_cost:
                    print(f"Epoch {epoch}, Sample {i}, Cost: {np.mean(cost)}")

            mean_cost = np.mean(epoch_costs)
            costs.append(mean_cost)

            if print_cost and epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {mean_cost}")
            if show_graph and epoch % 10 == 0:
                self.graph_cost(np.array(costs))

    def graph_cost(self, costs: np.ndarray):

        plt.plot(costs)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.title("Cost over epochs")
        plt.show()

    def one_hot_encode(self, y, num_classes):
        return np.eye(num_classes)[y.astype(int)].reshape(-1, num_classes)

    def __repr__(self):
        summary = ["Network Architecture:"]
        for i, layer in enumerate(self.layers):
            name = layer.name
            layer_type = layer.get_layer_type().name
            input_shape = layer.get_input_shape()
            output_shape = layer.get_output_shape()
            activation = layer.get_activation_function()
            activation_str = activation.name if activation else "None"

            summary.append(
                f"  ({i}) {name} | Type: {layer_type} | "
                f"Input: {input_shape} -> Output: {output_shape} | "
                f"Activation: {activation_str}"
            )
        return "\n".join(summary)


if __name__ == "__main__":

    # Cargar y preprocesar MNIST
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = X_train
    y_train = y_train

    # Normalizar y agregar canal
    X_train = X_train.astype(np.float64) / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)  # (batch, 28, 28, 1)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # One-hot encode
    network = Network(
        LayersConfig([])
    )  # temp instantiation just to access one_hot_encode
    Y_train = network.one_hot_encode(y_train, num_classes=10)

    # Crear red
    layers = [
        LayerConv(
            input_shape=(1, 28, 28, 1),
            specification=Specification_conv(
                c_filter=4,
                c_channels=1,
                c_filters=4,
                c_stride=2,
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
        LayerDense((16,), Activation_fn.RELU),
        LayerDense((10,), Activation_fn.SOFTMAX, name="output_layer"),
    ]

    network = Network(LayersConfig(layers))

    print(network)

    network.train(
        X=X_train,
        Y=Y_train,
        cost_function=loss_fn.CATEGORICAL_CROSSENTROPY,
        learning_rate=0.001,
        epochs=10,
        print_cost=True,
        show_graph=False,
    )
