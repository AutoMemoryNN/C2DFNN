import numpy as np
import matplotlib.pyplot as plt

from Layer import Layer, LAYER_TYPE, MomentumConfig, OptimizerConfig


class LayerDense(Layer):

    def __init__(self, input_shape: tuple[int], output_shape, activation, name):
        super().__init__(
            layer_type=LAYER_TYPE.DENSE, input_shape=input_shape, name=name
        )
        self.output_shape = output_shape
        self.activation = activation.lower()
        self.layers = []  # TODO: solve, layers by layer or all dense network

    def initialize_parameters(self) -> dict:
        """
        Inicializa los parámetros de la red.

        Argumentos:
        layers -- lista con las dimensiones de cada capa (incluyendo input y output)

        Retorna:
        parameters -- diccionario con parámetros "W1", "b1", ..., "WL", "bL"
                    Wl se inicializa con valores aleatorios pequeños y bl con ceros.
        """
        parameters = {}
        L = len(self.layers)
        for layer in range(1, L):
            parameters["W" + str(layer)] = (
                np.random.randn(self.layers[layer], self.layers[layer - 1]) * 0.01
            )
            parameters["b" + str(layer)] = np.zeros((self.layers[layer], 1))
        return parameters

    def forward_step(self, A_prev, W, b, activation):
        """
        Realiza un paso hacia adelante en una capa.

        Argumentos:
        A_prev -- activaciones de la capa anterior (shape: [n_prev, m])
        W -- matriz de pesos de la capa actual (shape: [n, n_prev])
        b -- vector de sesgos de la capa actual (shape: [n, 1])
        activation -- función de activación a aplicar ("sigmoid", "relu", "tanh", "softmax", "lineal")

        Retorna:
        A -- activaciones de la capa actual (shape: [n, m])
        cache -- tuple con (A_prev, W, b, activation, Z) para posible retropropagación o debugging
        """
        Z = W @ A_prev + b

        if activation.lower() == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
        elif activation.lower() == "relu":
            A = np.maximum(0, Z)
        elif activation.lower() == "tanh":
            A = np.tanh(Z)
        elif activation.lower() == "softmax":
            # Apoyo con IA
            # Para estabilidad numérica, se resta el máximo de Z en cada columna
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = expZ / np.sum(expZ, axis=0, keepdims=True)
        elif activation.lower() == "linear":
            A = Z
        else:
            raise ValueError("Función de activación desconocida: " + activation)

        cache = (A_prev, W, b, activation, Z)
        return A, cache

    def forward(self, X, parameters, activations):  # type: ignore
        """
        Realiza la propagación hacia adelante en la red.

        Argumentos:
        X -- datos de entrada (array de shape [n_x, m])
        parameters -- diccionario con los parámetros "W1", "b1", ..., "WL", "bL"
        activations -- lista de nombres de funciones de activación para cada capa
                    (por ejemplo, ["relu", "relu", "softmax"])

        Retorna:
        AL -- salida de la última capa
        caches -- lista de caches de cada capa para debugging o retropropagación
        """
        caches = []
        A = X
        L = len(activations)  # número de capas de la red

        for layer in range(1, L + 1):
            W = parameters["W" + str(layer)]
            b = parameters["b" + str(layer)]
            act_func = activations[layer - 1]
            A, cache = self.forward_step(A, W, b, act_func)
            caches.append(cache)

        AL = A
        return AL, caches

    def predict(self, X, parameters, activations):
        """
        Realiza la predicción de la red.

        Argumentos:
        X -- datos de entrada (array de shape [n_x, m])
        parameters -- diccionario de parámetros de la red
        activations -- lista de funciones de activación para cada capa

        Retorna:
        preds -- predicciones: si la última activación es softmax se retorna el índice
                con la mayor probabilidad para cada muestra; si es sigmoid se aplica un umbral de 0.6;
                en otro caso se retorna la salida directamente.
        """
        AL, _ = self.forward(X, parameters, activations)

        # Apoyo con IA
        last_activation = activations[-1].lower()
        if last_activation == "softmax":
            preds = np.argmax(AL, axis=0)
        elif last_activation == "sigmoid":
            preds = (AL > 0.6).astype(int)
        else:
            preds = AL  # para salidas lineales o de otro tipo
        return preds

    def accuracy(self, X, Y, parameters, activations):
        """
        Calcula la exactitud de la red.

        Argumentos:
        X -- datos de entrada (array de shape [n_x, m])
        Y -- etiquetas verdaderas (vector de shape [m,] o [1, m])
        parameters -- diccionario de parámetros de la red
        activations -- lista de funciones de activación para cada capa

        Retorna:
        acc -- exactitud de la predicción (porcentaje de aciertos)
        """
        preds = self.predict(X, parameters, activations)

        nCorrects = 0

        for i in range(len(Y)):
            # problem with shapes, Y(7352,) preds(561,) why?
            if Y[i] == preds[i] + 1:
                nCorrects += 1

        return nCorrects / len(Y)

    def mse(self, X, Y, parameters, activations):
        """
        Calcula el error cuadrático medio (MSE) entre la salida de la red y las etiquetas verdaderas.

        Argumentos:
        X -- datos de entrada (array de shape [n_x, m])
        Y -- etiquetas verdaderas (array, shape compatible con la salida de la red)
        parameters -- diccionario de parámetros de la red
        activations -- lista de funciones de activación para cada capa

        Retorna:
        error_mse -- valor del MSE
        """
        AL, _ = self.forward(X, parameters, activations)
        error_mse = np.mean((AL - Y) ** 2)
        return error_mse

    def load_y_data(self, file_path):
        """
        Lee el archivo de etiquetas y devuelve un array de NumPy.
        Cada línea tiene el formato: 'y: <etiqueta>'
        """
        labels = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # omite líneas vacías

                # Se puede quitar el prefijo 'y:' si existe
                if line.startswith("y:"):
                    line = line[len("y:") :].strip()

                try:
                    label = int(line)
                except ValueError:
                    label = float(line)
                labels.append(label)

        return np.array(labels)

    def cost(self, Yp, Y, costfunction):
        """
        Return the cost function.
        Arguments:
        Yp -- predictions
        Y -- true values
        cost – cost function “binary”, “multiclass”, “mse”
        Returns:
        cost -- ccost
        """
        m = Y.shape[0]
        costVal = 0
        if costfunction in ["sigmoid", "tanh"]:
            costVal = np.sum((Y - Yp) ** 2) / m
        elif costfunction in ["softmax"]:
            # J = - (1/m) * sum(sum(Y_ij * log(Yp_ij) for j in range(N)) for i in range(m))
            costVal = -np.sum(Y * np.log(Yp + 1e-8)) / m
        elif costfunction == ["lineal"]:
            costVal = np.sum((Y - Yp) ** 2) / m
        else:
            raise ValueError("Función de costo desconocida: " + costfunction)
        return costVal

    def backward_step(self, dA, cache: tuple, g: str) -> tuple:
        """
        Perform a backward step.

        Arguments:
        dA -- Gradient of activation (from the next layer)
        cache -- Tuple containing (A_prev, W, b, Z) from the forward pass
        g -- Activation function used ("sigmoid", "tanh", "softmax", "linear")

        Returns:
        dA_prev -- Gradient of activation from the previous layer
        dW -- Gradient of W (current layer)
        db -- Gradient of b (current layer)
        """

        A_prev, W, b, Z = cache

        m = A_prev.shape[1]

        dz = None
        dg = None

        if g == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            dg = s * (1 - s)
        elif g == "relu":
            dg = np.where(Z > 0, 1, 0)
        elif g == "tanh":
            dg = 1 - np.tanh(Z) ** 2
        elif g == "softmax":
            # Apoyo con IA
            dg = 1
        elif g == "linear":
            dg = 1
        else:
            raise ValueError("Función de activación desconocida: " + g)

        dz = dA * dg

        dW = dz @ A_prev.T / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dA_prev = W.T @ dz  # Gradient of the activation of the previous layer

        return dA_prev, dW, db

    # Ayuda: np.sum(…, axis=1, keepdims=True)
    def backward(self, AL, Y, caches, activations: list, cost_function) -> dict:  # type: ignore
        """
        Realiza la retropropagación de la red.

        Argumentos:
        AL -- salida de la última capa (predicciones), de forma (n_out, m)
        Y -- etiquetas verdaderas en one-hot encoding, de forma (n_out, m)
        caches -- lista de caches almacenados en la fase forward (uno por capa).
                Cada cache es una tupla (A_prev, W, b, Z).
        activations -- lista de funciones de activación utilizadas en cada capa
                    (ejemplo: ["relu", "relu", "softmax"])
        cost_function -- tipo de función de costo (“binary_crossentropy”, “categorical_crossentropy”, “mse”)

        Retorna:
        grads -- diccionario con los gradientes "dW1", "db1", ..., "dWL", "dbL"
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]

        A_prev_L, W_L, b_L, Z_L = caches[-1]

        if cost_function == "binary_crossentropy" and activations[-1] == "sigmoid":
            dAL = -((Y / AL) - ((1 - Y) / (1 - AL)))
            dZ_L = AL - Y
        elif (
            cost_function == "categorical_crossentropy" and activations[-1] == "softmax"
        ):
            dAL = AL - Y
            dZ_L = AL - Y
        elif cost_function == "mse" and activations[-1] == "linear":
            dAL = (AL - Y) / m
            dZ_L = AL - Y
        else:
            raise ValueError(
                "Error en la función de costo o activación: "
                + cost_function
                + " con "
                + activations[-1]
            )

        dW_L = dZ_L @ A_prev_L.T / m  # dW_L tendrá dimensión (n_out, n_prev)
        db_L = (
            np.sum(dZ_L, axis=1, keepdims=True) / m
        )  # db_L tendrá dimensión (n_out, 1)

        grads["dW" + str(L)] = dW_L
        grads["db" + str(L)] = db_L

        dA = dAL
        for layer in reversed(range(L - 1)):
            A_prev, W, b, Z = caches[layer]
            dA, dW, db = self.backward_step(dA, (A_prev, W, b, Z), activations[layer])
            grads["dW" + str(layer + 1)] = dW
            grads["db" + str(layer + 1)] = db

        return grads

    def update_parameters(self, optimizerConfig: OptimizerConfig | MomentumConfig):
        """
        Update parameters
        Arguments:
        parameters -- parameters "W1", "b1", “g1”,..., "WL", "bL", ”gL” (dictionary)
        grads -- gradients "dW1", "db1", ..., "dWL", "dbL" (dictionary)
        Returns:
        parameters -- parameters "W1", "b1", ..., "WL", "bL" (dictionary)
        """
        parameters = self.initialize_parameters()
        grads = np.zeros_like(parameters)

        L = len(parameters) // 2  # number of layers in the neural network

        for layer_index in range(L):
            parameters["W" + str(layer_index + 1)] = (
                parameters["W" + str(layer_index + 1)]
                - optimizerConfig.learning_rate * grads["dW" + str(layer_index + 1)]
            )
            parameters["b" + str(layer_index + 1)] = (
                parameters["b" + str(layer_index + 1)]
                - optimizerConfig.learning_rate * grads["db" + str(layer_index + 1)]
            )

        return parameters

    def train(
        self, X, Y, layers, activations, cost, learning_rate, iterations, print, graph
    ):
        """
        Train a neural network
        Arguments:
        X -- input data
        Y -- true values
        layers -- dimensions of each layer in our network (list)
        activations– activation activations of each layer (list) (“sigmoid”, “relu”, “tanh”,“softmax”, “lineal”)
        cost – cost function “binary”, “cathegorical”, “mse”
        learning_rate -- learning rate
        iterations -- number of iterations
        print-- print the cost every print iterations
        graph – graph the cost function for each iteration
        Returns:
        parameters -- parameters "W1", "b1", ..., "WL", "bL" (dictionary)
        """
        parameters = self.initialize_parameters()
        costs = []

        for i in range(iterations):
            AL, caches = self.forward(X, parameters, activations)
            cost_value = cost(AL, Y, cost)

            if i % print == 0:
                print(f"Cost after iteration {i}: {cost_value}")
                costs.append(cost_value)

            grads = self.backward(AL, Y, caches, activations, cost)
            parameters = self.update_parameters(learning_rate)

        if graph:

            plt.plot(costs)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Cost function over iterations")
            plt.show()

        return parameters

    def one_hot_encode(self, y, num_classes=5):
        """
        argument
            - x: a list of labels
            - num_classes: number of classes
        return
            - one hot encoding matrix (number of labels, number of class)
        """
        encoded = np.zeros((len(y), num_classes))

        for idx, val in enumerate(y):
            encoded[idx][val - 2] = 1

        return encoded

    class LayerDense(Layer):
        def __init__(self, input_shape: tuple[int], output_shape, activation, name):
            super().__init__(
                layer_type=LAYER_TYPE.DENSE, input_shape=input_shape, name=name
            )
