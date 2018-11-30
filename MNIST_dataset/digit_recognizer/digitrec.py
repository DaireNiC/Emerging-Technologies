# Author: Daire Ní Chathain
# Adapted from:
# (1): https://nextjournal.com/gkoehler/digit-recognition-with-keras
# (2): https://medium.com/@cafielo/build-a-handwritten-digit-recognition-model-with-keras-b8733274574c
# (3): https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb
# (4): https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
# (5): https://docs.python.org/3/distutils/packageindex.html#package-index

# Import keras.
import keras as kr
import os
import argparse
from utils import draw_digit
# import numpy
import numpy as np
from keras.datasets import mnist  # No need to re-invent the wheel
from keras.utils import np_utils
from keras.models import load_model

from PIL import Image
# constants
NUM_CLASSES = 10  # represents 0-9 digits
# where the model should be stored
MODEL_NAME = 'keras_mnist.h5'

"""
This function loads the MNIST dataset and preforms basic preprocessing on it
"""


def prep_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # building the input vector from the 28x28 pixels
    # in each dataset we use numpy to invert the values - so that the background is white and didgit is black
    # normalizing the data to speed up training
    X_train = ~np.array(
        list(X_train)).reshape(
        60000,
        784).astype(
            np.uint8) / 255
    X_test = ~np.array(list(X_test)).reshape(10000, 784).astype(np.uint8) / 255

    return X_train, y_train, X_test, y_test


"""
This function builds and trains a neural network using Keras and the MNIST dataset.
    INPUT: MNIST dataset --> tuple containing test & train images
    OUTPUT: Keras Sequential model
"""


def build_neural_net(MNIST_data):
    # unpack MNIST data
    X_train, y_train, X_test, y_test = MNIST_data
    # one-hot encoding using keras' numpy-related utilities
    Y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
    # Start a neural network, building it by layers.
    model = kr.models.Sequential()
    # Add a hidden layer with 600 neurons and an input layer with 784( 28 x 28
    # pixles)
    model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
    # We then stack another layer, this time one of 400 neurons.
    model.add(kr.layers.Dense(units=400, activation='relu'))
    # Add a ten neuron output layer to represent digits 0-9
    model.add(kr.layers.Dense(units=NUM_CLASSES, activation='softmax'))
    # Build the graph.
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    # train the neural net
    model.fit(
        X_train,
        Y_train,
        batch_size=120,
        epochs=1,
        verbose=2,
        validation_data=(
            X_test,
            Y_test))
    # evaluate score and print
    score = model.evaluate(X_test, Y_test, verbose=1)
    print(type(score[1]))
    print('Successfully created model with ' +
          score[1].astype(str) + ' accuracy')
    return model


"""
This function takes in an image and classifies the image using the neural network
    INPUT: Image file
"""


def predict_img(input_img):
    # Returns a compiled model
    model = load_model(MODEL_NAME)
    print('Classifying digit ...')
    # parsing the image to meet model input structure
    #img = np.invert(Image.open("test_img.png")).reshape(1,784)
    img = np.invert(Image.open(input_img)).reshape(1, 784)

    # get the corresponding categorical value from the hot encoded array
    # changes from binary array --> gets max (where the position of the 1 is)
    print(
        "Predicted: " +
        np.argmax(
            model.predict(img),
            axis=None,
            out=None).astype(str))


"""
This function takes in the keras sequential model and saves it to current dir
    INPUT: Keras Sequential model
"""


def save_model(model):
    # saving the model)
    model_path = os.path.join(MODEL_NAME)
    model.save(MODEL_NAME)
    print('Saved model as ' + MODEL_NAME)


def get_img(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def draw_input():
    try:
        draw_digit.open_canvas()
        predict_img('user_img.png')
    except BaseException:
        print("error creating digit drawings")


if __name__ == '__main__':
    # parse the command line arguments
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(description="digit recogniser")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i",
        "--image",
        action="store",
        help="Path to image you wish to classify",
        dest='image')
    group.add_argument("-d", "--draw", action="store_true",
                       help="Draw an a digit for recognition")

    args = parser.parse_args()
    if args is None:
        raise argparse.ArgumentTypeError('')

    if args.image:
        # if model exists already don't create a new one
        if (os.path.isfile(MODEL_NAME)):
            predict_img(args.image)
        # otherwise, create model and then classify
        else:
            print('No model found, creating model and then classifying')
            MNIST_data = prep_mnist_data()
            model = build_neural_net(MNIST_data)
            save_model(model)
            predict_img(args.image)
    if args.draw:
        draw_input()
