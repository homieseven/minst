from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2 # L2-regularisatio
from keras.models import model_from_yaml

import time
import sklearn
import numpy as np

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt

import os

ROOT_PATH = os.getcwd()

def main():
    #load YAML and create model
    yaml_file = open('model_99.57.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    model.load_weights("model_99.57.h5")
    print("Loaded model from disk")
    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])


    predict_df = pd.read_csv(os.path.join(ROOT_PATH, 'test.csv'))
    predict_data = predict_df.as_matrix().astype('float32') / 255
    predict_images = predict_data.reshape(predict_data.shape[0], 28, 28, 1)

    all_df = pd.read_csv(os.path.join(ROOT_PATH, 'train.csv'))
    (all_pixels_df, all_labels_df) = split_labels(all_df)

    all_pixel_data = all_pixels_df.as_matrix().astype('float32') /255
    all_images = all_pixel_data.reshape(all_pixel_data.shape[0], 28, 28, 1)

    global_images = np.append(predict_images, all_images, axis=0)
    mean_image = np.mean(global_images, axis=0)

    predict_images = (predict_images - mean_image)
    predict_labels = model.predict_classes(predict_images)

    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0,  # Randomly zoom image
        width_shift_range=3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=3,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images)

    predict_labels = np.zeros(predict_images.shape[0], dtype= np.int)
    number_of_aug_images = 10
    augumented_images = np.zeros((number_of_aug_images,28,28,1))
    augumented_labels = np.zeros(number_of_aug_images)
    for i in range(predict_images.shape[0]):
        augumented_classes = np.zeros(number_of_aug_images, dtype=np.int)
        image = predict_images[i,:]
        for j in range(number_of_aug_images):
            augumented_images[j,:] = train_datagen.random_transform(image)
        augumented_labels = model.predict_classes(augumented_images)
        for z in augumented_labels:
            augumented_classes[z] += 1
        predict_labels[i] = np.argmax(augumented_classes)

    # merge together
    idLabel = pd.Series(data=np.arange(1, predict_labels.size + 1), name='ImageId')
    serieLabel = pd.Series(data=predict_labels, name='Label')

    pr_df = pd.concat([idLabel, serieLabel], axis=1)
    path_for_prediction = os.path.join(ROOT_PATH, 'prediction.csv')
    result = pr_df.to_csv(path_for_prediction, index=False)




def split_labels(data: DataFrame):
    label = data['label']
    data = data.drop(columns='label')
    return (data, label)


if __name__ == '__main__':
    main()
    #example()