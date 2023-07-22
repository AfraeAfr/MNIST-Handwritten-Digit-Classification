# MNIST Handwritten Digit Classification
# This project implements a neural network model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

# Loading the MNIST data from keras.datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Unique values in Y_train and Y_test
print("Unique labels in training data:", np.unique(Y_train))
print("Unique labels in test data:", np.unique(Y_test))

# Data Preprocessing: Scaling the values
X_train = X_train / 255
X_test = X_test / 255

# Building the Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Compiling the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the Neural Network
model.fit(X_train, Y_train, epochs=10)

# Training data accuracy
_, training_accuracy = model.evaluate(X_train, Y_train)
print("Training data accuracy:", training_accuracy)

# Test data accuracy
_, test_accuracy = model.evaluate(X_test, Y_test)
print("Test data accuracy:", test_accuracy)

# Confusion Matrix
Y_pred = model.predict(X_test)
Y_pred_labels = [np.argmax(i) for i in Y_pred]
conf_mat = confusion_matrix(Y_test, Y_pred_labels)
plt.figure(figsize=(15, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Reds')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

# Predictive System Function
def predict_digit(image_path):
    input_image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    input_image_resize = cv2.resize(grayscale, (28, 28))
    input_image_resize = input_image_resize / 255
    image_reshaped = np.reshape(input_image_resize, [1, 28, 28])
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    return input_pred_label

# Predictive System
print("Enter the path of the image to be predicted:")
input_image_path = input()
input_pred_label = predict_digit(input_image_path)
print("The Handwritten Digit is recognized as:", input_pred_label)
