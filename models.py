import tensorflow as tf
import numpy as np


def create_mlp_model():
    """
    Creates a model for testing the inference.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=10, activation='softmax')

    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    return "MLP", model


def create_deep_mlp_model():
    """
    Creates a model for testing the inference.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=10, activation='softmax')

    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    return "Deep MLP", model


def create_conv2d_model():
    """
    Creates a model for testing the inference.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    return "Conv2D", model


def create_lstm_model():
    """
    Creates a model for testing the inference.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=64, input_shape=[28, 28]),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    return "LSTM", model


def create_stacked_lstm_model():
    """
    Creates a model for testing the inference.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=[28, 28]),
        tf.keras.layers.LSTM(units=64),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    return "Stacked LSTM", model














