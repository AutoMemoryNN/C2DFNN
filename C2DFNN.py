import time
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import cv2

from tensorflow.keras.datasets import mnist  # type: ignore
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from enum import Enum
from Layer import Layer, LAYER_TYPE, ACTIVATION_FN, OptimizerConfig, MomentumConfig
from LayerDense_v2 import LayerDense
from LayerConv import (
    LayerConv,
    LayerFlatten,
    LayerPooling,
    POOLING_FN,
    Specification_conv,
    Specification_pooling,
)


class LOSS_FN(Enum):
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
                if layer.get_layer_type() == LAYER_TYPE.CONVOLUTIONAL:
                    layer.name = f"conv_layer_{i} "
                elif layer.get_layer_type() == LAYER_TYPE.DENSE:
                    layer.name = f"dense_layer_{i}"
                elif layer.get_layer_type() == LAYER_TYPE.FLATTEN:
                    layer.name = f"flat_layer_{i} "
                elif layer.get_layer_type() == LAYER_TYPE.POOLING:
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

        if output_activation == ACTIVATION_FN.SOFTMAX:
            Yp_labels = np.argmax(Yp, axis=1)
            Y_true = np.argmax(Y, axis=1)
        elif output_activation == ACTIVATION_FN.SIGMOID:
            Yp_labels = (Yp > 0.5).astype(int)
            Y_true = Y.astype(int)
        else:
            raise ValueError(
                f"Accuracy not implemented for activation: {output_activation}"
            )

        acc = np.mean(Yp_labels == Y_true)
        return acc

    def cost(self, Yp: np.ndarray, Y: np.ndarray, costfunction: LOSS_FN) -> np.ndarray:
        output_activation = self.layers[-1].get_activation_function()

        # clipped values to avoid log(0)
        if (
            costfunction == LOSS_FN.CATEGORICAL_CROSSENTROPY
            and output_activation == ACTIVATION_FN.SOFTMAX
        ):
            Yp_clipped = np.clip(Yp, 1e-15, 1 - 1e-15)
            return -np.sum(Y * np.log(Yp_clipped), axis=1)
        elif (
            costfunction == LOSS_FN.MEAN_SQUARED_ERROR
            and output_activation == ACTIVATION_FN.RELU
        ):
            return np.mean(np.square(Y - Yp), axis=1)
        elif (
            costfunction == LOSS_FN.BINARY_CROSSENTROPY
            and output_activation == ACTIVATION_FN.SIGMOID
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
        self, Yp: np.ndarray, Y: np.ndarray, costfunction: LOSS_FN
    ) -> np.ndarray:
        if costfunction == LOSS_FN.CATEGORICAL_CROSSENTROPY:
            return Yp - Y
        elif costfunction == LOSS_FN.MEAN_SQUARED_ERROR:
            return 2 * (Yp - Y) / Y.shape[0]
        elif costfunction == LOSS_FN.BINARY_CROSSENTROPY:
            Yp_clipped = np.clip(Yp, 1e-15, 1 - 1e-15)
            return -Y / Yp_clipped + (1 - Y) / (1 - Yp_clipped)
        else:
            raise ValueError(
                f"Unsupported cost function: {costfunction}. Supported functions are: {list(LOSS_FN)}."
            )

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cost_function: LOSS_FN,
        optimizerConfig: OptimizerConfig = OptimizerConfig(learning_rate=0.001),
        epochs: int = 50,
        batch_size: int = 16,
        print_cost=False,
        show_graph=False,
        X_test: np.ndarray | None = None,
        Y_test: np.ndarray | None = None,
    ):
        costs = []
        self.train_accuracies = []

        for epoch in range(epochs):
            start_time = time.time()
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

                batch_cost = np.mean(cost)

                epoch_costs.append(batch_cost)

                # Backward
                dA = self.d_cost(Z, y, cost_function)
                for layer in reversed(self.layers):
                    dA = layer.backward(dA)
                    self.parameters[layer.name] = layer.get_parameters()

                # Update
                for layer in self.layers:
                    layer.update_parameters(optimizerConfig=optimizerConfig)

                if i % 10 == 0 and print_cost:
                    print(
                        f"[Epoch {epoch + 1}/{epochs}] Batch {i + 1}/{(X.shape[0] + batch_size - 1) // batch_size} | "
                        f"Cost: {batch_cost:.6f}"
                    )

            # each epoch
            mean_cost = np.mean(epoch_costs)
            acc = self.accuracy(X, Y)
            costs.append(mean_cost)
            self.train_accuracies.append(acc)
            epoch_time = time.time() - start_time

            if print_cost:
                print(
                    f"[Epoch {epoch+1}/{epochs}] Cost: {mean_cost:.6f} | "
                    f"Train Accuracy: {acc * 100:.2f}% | Time: {epoch_time:.2f}s"
                )

        if show_graph:
            self.graph_cost(np.array(costs), np.array(self.train_accuracies))

        # Comparar contra test si se pasa
        if X_test is not None and Y_test is not None:
            test_acc = self.accuracy(X_test, Y_test)
            print(
                f"\nFinal Train Accuracy: {self.train_accuracies[-1] * 100:.2f}%\n"
                f"Test Accuracy:         {test_acc * 100:.2f}%"
            )

    def graph_cost(self, costs: np.ndarray, accuracies: np.ndarray):
        epochs = np.arange(1, len(costs) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cost", color="tab:blue")
        ax1.plot(epochs, costs, marker="o", color="tab:blue", label="Cost")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        best_epoch = np.argmin(costs) + 1
        ax1.axvline(
            x=float(best_epoch),
            color="red",
            linestyle="--",
            label=f"Min Cost (Epoch {best_epoch})",
        )
        ax1.scatter(best_epoch, costs[best_epoch - 1], color="red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="tab:green")
        ax2.plot(
            epochs, accuracies * 100, marker="s", color="tab:green", label="Accuracy"
        )
        ax2.tick_params(axis="y", labelcolor="tab:green")

        fig.tight_layout()
        plt.title("Training Progress: Cost and Accuracy")
        fig.legend(loc="upper right", bbox_to_anchor=(1, 0.85))
        plt.grid(True)
        plt.show()

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


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.astype(int)].reshape(-1, num_classes)


# IA Generated
def reduce_mnist_by_category(X, y, reduce_fraction=0.5, target_size=(14, 14)):
    selected_X = []
    selected_y = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        keep_n = int(len(idx) * reduce_fraction)
        selected_idx = idx[:keep_n]
        selected_X.append(X[selected_idx])
        selected_y.append(y[selected_idx])

    reduced_X = np.concatenate(selected_X)
    reduced_y = np.concatenate(selected_y)

    # Mezclar los datos
    perm = np.random.permutation(len(reduced_y))
    reduced_X = reduced_X[perm]
    reduced_y = reduced_y[perm]

    # Redimensionar imágenes (por ejemplo, a 14x14)
    resized_X = np.array([cv2.resize(img, target_size) for img in reduced_X])
    resized_X = resized_X.astype(np.float64) / 255.0
    resized_X = resized_X.reshape(-1, target_size[0], target_size[1], 1)

    return resized_X, reduced_y


if __name__ == "__main__":
    (X_raw, y_raw), (_, _) = mnist.load_data()

    # Reduce MNIST
    X_reduced, y_reduced = reduce_mnist_by_category(
        X_raw, y_raw, reduce_fraction=0.5, target_size=(14, 14)
    )

    # 85% train, 15% test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_reduced, test_size=0.15, stratify=y_reduced, random_state=42
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Define layers
    layers = [
        LayerConv(
            input_shape=(1, 14, 14, 1),
            specification=Specification_conv(
                c_filter=4,
                c_channels=1,
                c_filters=4,
                c_stride=2,
                c_pad=0,
                activation=ACTIVATION_FN.RELU,
            ),
        ),
        LayerPooling(
            specification=Specification_pooling(
                p_filter=2, p_stride=2, p_function=POOLING_FN.MAX
            ),
        ),
        LayerFlatten(),
        LayerDense((16,), ACTIVATION_FN.RELU),
        LayerDense((10,), ACTIVATION_FN.SOFTMAX, name="output_layer"),
    ]

    network = Network(LayersConfig(layers))

    print(network)

    network.train(
        X=X_train,
        Y=one_hot_encode(y_train, num_classes=10),
        cost_function=LOSS_FN.CATEGORICAL_CROSSENTROPY,
        optimizerConfig=MomentumConfig(learning_rate=0.005, momentum=0.75),
        epochs=10,
        print_cost=True,
        show_graph=True,
        X_test=X_test,
        Y_test=one_hot_encode(y_test, num_classes=10),
    )
