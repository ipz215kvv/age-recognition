import cv2


IMAGE_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)


def process_image(image, size=IMAGE_SIZE):
    resized = cv2.resize(image, size)
    return resized / 255.0
