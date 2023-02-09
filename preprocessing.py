import numpy as np
import cv2
NUM_CLASSES = 10

def shapify_image(train_data, img_size=(180, 180)):
    """
    Reshape the image to img_size
    """
    X_train, Y_train = train_data
    X_train = np.array([[cv2.resize(img, img_size)] for img in X_train])

    return (np.array(X_train), np.array(Y_train)[:,np.newaxis])

def preprocess_image(train_data):
    """
    Preprocess the image
    """
    X_train, Y_train = train_data
    X_train = 255 - X_train
    X_train[X_train < 100] = 0.
    X_train[X_train >= 100] = 1.


    Y_train = one_hot_encode(Y_train)

    return (X_train, Y_train)

def one_hot_encode(Y_train, num_classes=NUM_CLASSES):
    """
    One hot encode the labels
    """
    Y_train = np.eye(num_classes)[Y_train.reshape(-1)]
    return Y_train
