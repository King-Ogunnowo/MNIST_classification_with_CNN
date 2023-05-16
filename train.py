import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import(
    train_test_split
)

def build_network_architecture():
    """
    This function creates and compiles the network architecture. 
    The network is a CNN to be trained and used to predict MNIST data instances. 
    
    NETWORK ARCHITECTURE
    --------------------
    
    The architecture is quite simple. 
        - the input layer creates 64 filters, each having a 7 x 7 shape, makes use of zero-padding and takes in images of input shape 28 height by 28 depth by 1 color channel (grayscale)
        - the hidden layers consist of Convolutional layers, combined with MaxPooling layers. The Conv. layers output feature maps which are then subsampled by the MaxPooling Layer. The Neurons in the Maxpooling layer simply selects the highest value in its receptive layer, shrinking the image for the next layer.
        - Stack of Dense Layers. The stack of Dense layers take the output of the flatten layer, and pass unto the output layer with has 10 outputs, and uses a soft max function 
        
    COMPILING THE MODEL
    -------------------
    
    The model is compiled with the AdamW optiizer, which is quite popular with CNNs, I am currently researching on why.
    The sparse categorical cross entropy loss is used, since we are dealing with a multi class problem.
    The metric used here is accuracy. You are free to use other metrics as well
    """
    ConvNet = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, 7, 
                activation = "relu", 
                padding = "same", 
                input_shape = [28, 28, 1]
            ),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(
                128, 
                3,
                activation = "relu",
                padding = "same"
            ),
            tf.keras.layers.Conv2D(
                128,
                3,
                activation = "relu",
                padding = "same"
            ),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(
                256,
                3, 
                activation = "relu", 
                padding = "same"
            ),
            tf.keras.layers.Conv2D(
                256,
                3,
                activation = "relu",
                padding = "same"
            ),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                128,
                activation = "relu"
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                64, 
                activation = "relu"
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                10, 
                activation = "softmax"
            )
        ]
    )
    ConvNet.compile(
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate = 0.0001
    ),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = "accuracy"
    )
    return ConvNet

def process_MNIST_data():
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, 
        y_train_full, 
        test_size = 0.25, 
        stratify = y_train_full # --- stratify, so we have all classes equally represented. Similar to stratified sampling
    )
    X_train = X_train / 225.0
    X_valid = X_valid / 225.0
    X_test = X_test / 225.0
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train():
    ConvNet = build_network_architecture()
    X_train, X_valid, X_test, y_train, y_valid, y_test = process_MNIST_data()
    history = ConvNet.fit(
        X_train, 
        y_train, 
        epochs = 10, 
        validation_data = (
            X_valid, 
            y_valid
        )
    )
    ConvNet.save('model/ConvNet')
    return history

def plot_performance(peformance):
    plt.figure(figsize = (15, 10))
    sns.lineplot(peformance.history)
    plt.savefig('performance.png')


history = train()
plot_performance(history)
    
    

