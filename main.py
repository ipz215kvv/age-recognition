from camera import Camera


if __name__ == "__main__":
    with Camera() as camera:
        for face in camera.faces:
            face.text = "The face!"
