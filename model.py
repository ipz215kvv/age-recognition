import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tensorflow.keras.models import Sequential, load_model, Model as TfModel
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from utils import process_image, INPUT_SHAPE


class Model:
    def __init__(self, base_model=None):
        if base_model:
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(64, activation="relu")(x)
            x = Dropout(0.5)(x)
            predictions = Dense(1)(x)
            self.model = TfModel(inputs=base_model.input, outputs=predictions)
        else:
            self.model = None

    def load_from_dataset(
        self,
        dataset_path="./UTKFace/",
        save_path="model.keras",
    ):
        dataset_path = Path(dataset_path)
        X, y = self.load_data(dataset_path)

        if not self.model:
            self.model = Sequential(
                [
                    Input(INPUT_SHAPE),
                    Conv2D(32, (3, 3), activation="relu"),
                    MaxPooling2D(2, 2),
                    Conv2D(64, (3, 3), activation="relu"),
                    MaxPooling2D(2, 2),
                    Conv2D(128, (3, 3), activation="relu"),
                    MaxPooling2D(2, 2),
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(1),
                ]
            )
        self.model = self.train(X, y, save_path)
        self.model.save(save_path)

    def load_from_model(self, model_path="model.keras"):
        self.model = load_model(model_path)

    def filename_to_info(self, filename):
        extension = ".jpg.chip.jpg"
        info = filename[: -len(extension)]
        age, gender, race, timestamp = info.split("_")
        timestamp_format = "%Y%m%d%H%M%S%f"
        timestamp = timestamp + "000"
        return (
            int(age),
            int(gender),
            int(race),
            datetime.strptime(timestamp, timestamp_format),
        )

    def load_data(self, dataset_path):
        file_list = [file for file in dataset_path.iterdir() if file.is_file()]

        def process_file(filename):
            try:
                age, _, _, _ = self.filename_to_info(filename.name)
                image = cv2.imread(str(filename))
                image = process_image(image)
                return image, age
            except Exception as e:
                return None

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_file, file_list))

        results = [obj for obj in results if obj is not None]
        images, ages = zip(*results)

        return np.array(images), np.array(ages)

    def train(self, X, y, save_path):
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        self.model.summary()
        self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[ModelCheckpoint(save_path, save_best_only=True)],
        )
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return self.model

    def predict(self, image_matrix):
        assert hasattr(self, "model")

        return self.model.predict(image_matrix)[0][0]
