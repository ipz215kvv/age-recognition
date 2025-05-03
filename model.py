import os, cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from utils import proccess_image


class Model:
    def load_from_dataset(
        self,
        dataset_path="./UTKFace Dataset/",
        save_path="model.keras",
    ):
        X, y = self.load_data(dataset_path)
        self.model = self.train(X, y, save_path)
        self.model.save(save_path)

    def load_from_model(self, model_path="model.keras"):
        self.model = load_model(model_path)

    def filename_to_info(self, filename):
        info = Path(filename).stem
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
        images = []
        ages = []

        for filename in os.listdir(dataset_path):
            try:
                age, _, _, _ = self.filename_to_info(filename)
                ages.append(age)

                image_path = os.path.join(dataset_path, filename)
                image = cv2.imread(image_path)
                image = proccess_image(image)
                images.append(image)
            except Exception:
                continue

        images = np.array(images)
        ages = np.array(ages)
        return images, ages

    def train(self, X, y, save_path):
        model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
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

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        model.summary()
        model.fit(
            X,
            y,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            callbacks=[ModelCheckpoint(save_path, save_best_only=True)],
        )
        return model

    def predict(self, image_matrix):
        assert hasattr(self, "model")

        return self.model.predict(image_matrix)[0][0]
