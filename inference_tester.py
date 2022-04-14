import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # Disable GPU
DEVICE = 'GPU'
import tensorflow as tf
import numpy as np
import models

print("Tensorflow version: ", tf.__version__)
print("loading Data")
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = tf.keras.utils.to_categorical(data['y_train'])
    test_examples = data['x_test']
    test_labels = tf.keras.utils.to_categorical(data['y_test'])

print("Data loaded")





def train_model(label, model):
    print('Evaluating Model: ', label)

    print(f"({DEVICE}) Timing --- Training model")
    start = time.time()
    model.fit(train_examples, train_labels, epochs=5)
    end = time.time()
    print(f"({DEVICE}) Training time: ", end - start)
    print(f"({DEVICE}) Training Inference: ", ((end - start) / len(train_examples) / 5))

    print(f"({DEVICE}) Timing --- Evaluating model")
    start = time.time()
    print(f"({DEVICE}) Scores: ", model.evaluate(test_examples, test_labels))
    end = time.time()
    print(f"({DEVICE}) Evaluation time: ", end - start)
    print(f"({DEVICE}) Evaluation Inference: ", (end - start) / len(test_examples))

if __name__ == '__main__':
    function_names = [func for func in dir(models) if func.startswith('create_')]
    print('Available functions: ', function_names)

    for model_name in function_names:
        model_func = getattr(models, model_name)
        label, model = model_func()
        model.summary()
        # train_model(label, model)
        print()

