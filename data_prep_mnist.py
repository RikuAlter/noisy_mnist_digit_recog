import pickle as pk
import numpy as np
from keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt


def load_data(path):
    data_file = open(path, "rb")
    raw_data = pk.load(data_file)
    image_data = np.array(raw_data[0])
    label_data = np.array(raw_data[1])
    return image_data, label_data


def normalize_image_list(images):
    processed_images = np.copy(images)
    for i in range(len(images)):
        processed_images[i] = (processed_images[i] - processed_images[i].min()) / processed_images[i].max()
    return processed_images


def categorize_labels(labels):
    cat_labels = to_categorical(labels)
    return cat_labels


def blur_images(images):
    for i in range(len(images)):
        images[i] = cv2.GaussianBlur(images[i], (3,3), 1)#cv2.medianBlur(images[i], 3)
    return images

def equalize_histogram(images):
    for i in range(len(images)):
        image = (images[i]*255).astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3,3))
        image = clahe.apply(image)
        images[i] = image/255
    return images


def preprocess_data(images, labels):
    processed_images = normalize_image_list(images)
    processed_images = blur_images(processed_images)
    processed_images = equalize_histogram(processed_images)
    processed_images = processed_images.reshape((-1, 28, 28, 1))
    processed_labels = categorize_labels(labels)
    return processed_images, processed_labels


def load_and_process_data(path):
    images, labels = load_data(path)
    return preprocess_data(images, labels)
