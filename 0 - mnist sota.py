from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2 # L2-regularisation

import time
import sklearn
import numpy as np

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt

import os

ROOT_PATH = os.getcwd()

def main():
    predict_df = pd.read_csv(os.path.join(ROOT_PATH, 'test.csv'))
    predict_data = predict_df.as_matrix().astype('float32') / 255.0
    predict_images = predict_data.reshape(predict_data.shape[0], 28, 28, 1)

    all_df = pd.read_csv(os.path.join(ROOT_PATH, 'train.csv'))
    all_df = sklearn.utils.shuffle(all_df)
    (all_pixels_df, all_labels_df) = split_labels(all_df)

    all_pixel_data = all_pixels_df.as_matrix().astype('float32')/ 255.0
    all_labels = all_labels_df.as_matrix()
    all_labels = to_categorical(all_labels,10)
    all_images = all_pixel_data.reshape(all_pixel_data.shape[0], 28, 28, 1)

    global_images = np.concatenate((all_images, predict_images), axis=0)
    mean_image = np.mean(global_images, axis=0)

    predict_images = (predict_images - mean_image)
    all_images = (all_images - mean_image)


    # split to train, val, test
    train_data = all_images[:39000]
    train_label = all_labels[:39000]
    val_data = all_images[39000:41000]
    val_label = all_labels[39000:41000]
    test_data = all_images[41000:42000]
    test_label = all_labels[41000:42000]


    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    #model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(32, (3, 3)))
    #model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    #model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(64, (3, 3)))
    #model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    # Fully connected layer
    model.add(layers.Dense(512))
    #model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Dense(10, activation='softmax'))


    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])


    start_time = time.time()
    history = model.fit(x=train_data, y=train_label, batch_size=16, shuffle=True, validation_data=(val_data, val_label), epochs=10)
    training_time = time.time() - start_time
    print("Training time", training_time)
    predict_labels = model.predict_classes(predict_images)

    # print history
    history_dict = history.history
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    epochs = range(1, len(acc_values) + 1)
    # print test result
    result = model.evaluate(test_data, test_label)
    print(result)

    plt.clf()
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print(result)




def split_labels(data: DataFrame):
    label = data['label']
    data = data.drop(columns='label')
    return (data, label)




if __name__ == '__main__':
    main()
    #example()