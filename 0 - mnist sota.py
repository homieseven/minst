from keras import layers
from keras import models
from keras.utils import to_categorical

import time
import sklearn
import numpy as np

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt

import os

ROOT_PATH = os.getcwd()
NUMBER_OF_EPOCHS = 30
BATCH_SIZE = 64

class ModelContainer:

    def __init__(self, model = None, description = None, text_color = None, train_acc = None, val_acc = None, training_time = None, result = None):
        self.model = model
        self.description = description
        self.text_color = text_color
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.training_time = training_time
        self.result = result


def main():

    (train_data, train_label, val_data, val_label, test_data, test_label) = prepare_data()
    containers = prepare_model_containers()

    for model_container in containers:

        model = model_container.model
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        #train
        start_time = time.time()
        history = model.fit(x=train_data, y=train_label, batch_size=BATCH_SIZE, shuffle=True, validation_data=(val_data, val_label), epochs=NUMBER_OF_EPOCHS, verbose=2)
        training_time = time.time() - start_time

        #evaluate
        result = model.evaluate(test_data, test_label)

        history_dict = history.history
        model_container.train_acc = history_dict['acc']
        model_container.val_acc = history_dict['val_acc']
        model_container.training_time = training_time

        # print test result
        model_container.result = result

    x = range(1,NUMBER_OF_EPOCHS+1)

    legend_titles = []
    for model_container in containers:
        plt.plot(x, model_container.train_acc, model_container.text_color)
        legend_titles.append('Train acc {}'.format(model_container.description))
        plt.plot(x, model_container.val_acc, '{}o'.format(model_container.text_color))
        legend_titles.append('Val acc {}'.format(model_container.description))
        print("Model: %s, results: %s, time: %d" % (model_container.description, model_container.result, model_container.training_time))

    # Shrink current axis by 20%
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(legend_titles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def prepare_data():
    predict_df = pd.read_csv(os.path.join(ROOT_PATH, 'test.csv'))
    predict_data = predict_df.as_matrix().astype('float32') / 255.0
    predict_images = predict_data.reshape(predict_data.shape[0], 28, 28, 1)

    all_df = pd.read_csv(os.path.join(ROOT_PATH, 'train.csv'))
    all_df = sklearn.utils.shuffle(all_df)
    (all_pixels_df, all_labels_df) = split_labels(all_df)

    all_pixel_data = all_pixels_df.as_matrix().astype('float32') / 255.0
    all_labels = all_labels_df.as_matrix()
    all_labels = to_categorical(all_labels, 10)
    all_images = all_pixel_data.reshape(all_pixel_data.shape[0], 28, 28, 1)

    global_images = np.concatenate((all_images, predict_images), axis=0)
    mean_image = np.mean(global_images, axis=0)

    predict_images = (predict_images - mean_image)
    all_images = (all_images - mean_image)

    # split to train, val, test
    train_data = all_images[:36000]
    train_label = all_labels[:36000]
    val_data = all_images[36000:39000]
    val_label = all_labels[36000:39000]
    test_data = all_images[39000:42000]
    test_label = all_labels[39000:42000]
    return (train_data, train_label, val_data, val_label, test_data, test_label)

def split_labels(data: DataFrame):
    label = data['label']
    data = data.drop(columns='label')
    return (data, label)


def prepare_model_containers():

    containers = []

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(10, activation='softmax'))
    mc = ModelContainer(model = model, description = "32(f3,1)/MP(2)/32(f3,1)/1024", text_color='b')
    containers.append(mc)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (5, 5)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(10, activation='softmax'))
    mc = ModelContainer(model=model, description="32(f3,1)/MP(2)/32(f5,1)/1024", text_color='r')
    containers.append(mc)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), input_shape=(28, 28, 1)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (5, 5)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(10, activation='softmax'))
    mc = ModelContainer(model=model, description="32(f5,1)/MP(2)/32(f5,1)/1024", text_color='g')
    containers.append(mc)

    return containers


# 1 - 0.046999341943479524, 0.989
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
# #model.add(layers.normalization.BatchNormalization())
# model.add(layers.Activation("relu"))
#
# model.add(layers.Conv2D(32, (3, 3)))
# #model.add(layers.normalization.BatchNormalization())
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Conv2D(64, (3, 3)))
# #model.add(layers.normalization.BatchNormalization())
# model.add(layers.Activation("relu"))
#
# model.add(layers.Conv2D(64, (3, 3)))
# #model.add(layers.normalization.BatchNormalization())
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Flatten())
# # Fully connected layer
# model.add(layers.Dense(512))
# #model.add(layers.normalization.BatchNormalization())
# model.add(layers.Activation("relu"))
#
# model.add(layers.Dense(10, activation='softmax'))

# 2 - 0.03971657040917034, 0.994
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Conv2D(32, (3, 3)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Conv2D(64, (3, 3)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Conv2D(64, (3, 3)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Flatten())
# # Fully connected layer
# model.add(layers.Dense(1024))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Dense(10, activation='softmax'))

# 3 - 0.01563, 0.994
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Conv2D(64, (3, 3)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Flatten())
# # Fully connected layer
# model.add(layers.Dense(1024))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Dense(10, activation='softmax'))

# 4 -  0.09612 - 0.98
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Flatten())
# # Fully connected layer
# model.add(layers.Dense(1024))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Dense(10, activation='softmax'))


# 5 - 0.0604024, 0.989
# model = models.Sequential()
# model.add(layers.Conv2D(32, (5, 5), input_shape=(28, 28, 1)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Flatten())
# # Fully connected layer
# model.add(layers.Dense(1024))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Dense(10, activation='softmax'))

# 6 - 0.072134, 0.99
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Conv2D(64, (3, 3)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Flatten())
# # Fully connected layer
# model.add(layers.Dense(1024))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Dense(10, activation='softmax'))

# 7 - 0.042227, 0.991
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Conv2D(64, (3, 3)))
# model.add(layers.Activation("relu"))
#
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Flatten())
# # Fully connected layer
# model.add(layers.Dense(1024))
# model.add(layers.Activation("relu"))
#
# model.add(layers.Dense(10, activation='softmax'))
#
#

if __name__ == '__main__':
    main()
    #example()