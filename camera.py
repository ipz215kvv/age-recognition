import cv2
import numpy as np

from utils import proccess_image


def esc_pressed():
    return cv2.waitKey(1) & 0xFF == 27


class Face:
    def __init__(self, image):
        self.image = image
        self.is_highlighted = True
        self.text = None

    @property
    def as_matrix(self):
        image = proccess_image(self.image)
        return np.expand_dims(image, axis=0)


class Camera:
    def __init__(self, window_title="Camera", device_index=0):
        self.window_title = window_title
        self.device_index = device_index

    def __enter__(self):
        filename = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(filename)
        self.capture = cv2.VideoCapture(self.device_index)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capture.release()
        cv2.destroyAllWindows()

    @property
    def faces(self):
        while True:
            is_success, frame = self.capture.read()
            if not is_success:
                break

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detected_faces = self.face_cascade.detectMultiScale(
                gray_image, scaleFactor=1.2, minNeighbors=5
            )

            for x, y, w, h in detected_faces:
                image = frame[y : y + h, x : x + w]
                face = Face(image)
                yield face

                color = (255, 0, 0)
                thickness = 2
                font = cv2.FONT_HERSHEY_DUPLEX
                if face.is_highlighted:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                if face.text:
                    cv2.putText(
                        frame,
                        face.text,
                        (x, y - 10),
                        font,
                        0.9,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )

            cv2.imshow(self.window_title, frame)

            if esc_pressed():
                break
