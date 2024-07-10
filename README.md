
Image dimensionality reduction project that uses neural networks with auto encoders for feature extraction and image compression
# AutoEncoder for Image Reconstruction using CIFAR-10

This project implements an AutoEncoder model for image reconstruction using the CIFAR-10 dataset. The model compresses the input images into a lower-dimensional representation and then reconstructs the images from this representation. The goal is to achieve a high-quality reconstruction of the original images.

## Overview

An AutoEncoder is a type of artificial neural network used to learn efficient codings of unlabeled data. In this project, the AutoEncoder is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. The model is composed of two parts:
- **Encoder**: Compresses the input images into a latent-space representation.
- **Decoder**: Reconstructs the images from the latent-space representation.

## Features

- Implementation of an AutoEncoder using TensorFlow and Keras.
- Uses convolutional layers for both the encoder and decoder.
- Normalizes image data for better performance.
- Visualizes original and reconstructed images for comparison.

## Setup

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/autoencoder-image-reconstruction.git
    cd autoencoder-image-reconstruction
    ```

2. Install the required packages:
    ```bash
    pip install tensorflow numpy matplotlib
    ```

## Usage

### Training the Model

The following script loads the CIFAR-10 dataset, preprocesses the data, defines the AutoEncoder model, and trains it.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load and preprocess the data
(x_train, _), (x_test, _) = cifar10.load_data()

# Normalize the data to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the AutoEncoder model for image data
class ImageAutoEncoder(Model):
    def __init__(self):
        super(ImageAutoEncoder, self).__init__()
        self.encoder = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same')
        ])
        self.decoder = Sequential([
            Conv2DTranspose(16, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Create an instance of the model
image_auto_encoder = ImageAutoEncoder()

# Compile the model
image_auto_encoder.compile(optimizer='adam', loss='mse')

# Train the model
history = image_auto_encoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

# Visualize the results
def plot_results(x_test, decoded_imgs, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        plt.axis("off")

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()

# Predict on the test data
decoded_imgs = image_auto_encoder.predict(x_test)

# Plot the results
plot_results(x_test, decoded_imgs)
