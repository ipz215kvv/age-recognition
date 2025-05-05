import cv2


def process_image(image, size=(64, 64)):
    resized = cv2.resize(image, size)
    return resized / 255.0
