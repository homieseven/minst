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

    # save in the same format as all data

    predict_df.insert(loc=0, column="label", value=predict_labels)
    path_for_fake_train = os.path.join(ROOT_PATH, 'fake_train.csv')
    result = predict_df.to_csv(path_for_fake_train, index=False)




def split_labels(data: DataFrame):
    label = data['label']
    data = data.drop(columns='label')
    return (data, label)


if __name__ == '__main__':
    main()
    #example()