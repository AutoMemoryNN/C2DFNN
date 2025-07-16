import numpy as np
import cv2
from tensorflow.keras.datasets import mnist  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.optimizers import SGD  # type: ignore
from sklearn.model_selection import train_test_split

from C2DFNN import reduce_mnist_by_category, one_hot_encode


(X_raw, y_raw), (_, _) = mnist.load_data()
X_reduced, y_reduced = reduce_mnist_by_category(
    X_raw, y_raw, reduce_fraction=0.5, target_size=(14, 14)
)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y_reduced, test_size=0.15, stratify=y_reduced, random_state=42
)

Y_train = one_hot_encode(y_train, 10)
Y_test = one_hot_encode(y_test, 10)

model = Sequential(
    [
        Conv2D(
            filters=4,
            kernel_size=(4, 4),
            strides=1,
            activation="relu",
            input_shape=(14, 14, 1),
        ),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(16, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)


optimizer = SGD(learning_rate=0.005, momentum=0.75)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)


model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=16,
    verbose=1,
)
_, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test accuracy: {accuracy * 100:.2f}%")
