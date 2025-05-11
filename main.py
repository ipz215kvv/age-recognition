from tensorflow.keras.applications import EfficientNetB0

from camera import Camera
from model import Model
from utils import INPUT_SHAPE


def test_model():
    model_path = input("Enter model path: ")
    model = Model()
    model.load_from_model(model_path)

    with Camera() as camera:
        for face in camera.faces:
            age = model.predict(face.as_matrix)
            face.text = f"Age: {int(age)}"


def train_model():
    dataset_path = input("Enter dataset path: ")
    save_path = input("Enter save path for a model(with `.keras` extension): ")
    model = Model()
    model.load_from_dataset(dataset_path, save_path)
    print("Model trained successfuly")


def transfer_model():
    dataset_path = input("Enter dataset path: ")
    save_path = input("Enter save path for a model(with `.keras` extension): ")
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=INPUT_SHAPE
    )
    model = Model(base_model)
    model.load_from_dataset(dataset_path, save_path)
    print("Model transfered successfuly")


if __name__ == "__main__":
    while True:
        options = {
            "1": test_model,
            "2": train_model,
            "3": transfer_model,
        }
        print(
            """
O P T I O N S:
[1] Test a model with your camera
[2] Train your model
[3] Transfer to existing model
            """
        )
        entered_option = input("Select your option(write its number): ")
        if not (entered_option.isdigit() and entered_option in list(options.keys())):
            print("You've entered invalid option")
            continue

        options[entered_option]()
