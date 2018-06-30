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

    val_data = all_images[39000:42000]
    val_label = all_labels[39000:42000]

    test_data = all_images[39000:42000]
    test_label = all_labels[39000:42000]

    # 99.1/98.86
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(layers.Flatten())
    # # Fully connected layer
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(10, activation='softmax'))

    # 99.36 / 99.59 / 99.56(99.7) TOP
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(layers.normalization.BatchNormalization(axis=-1))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.normalization.BatchNormalization(axis=-1))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.normalization.BatchNormalization(axis=-1))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(layers.Flatten())
    # # Fully connected layer
    # model.add(layers.normalization.BatchNormalization())
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.normalization.BatchNormalization())
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(10, activation='softmax'))



    #99.6
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), use_bias=False, input_shape=(28, 28, 1)))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(32, (3, 3),  use_bias=False))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3),  use_bias=False))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(64, (3, 3),  use_bias=False))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    # Fully connected layer
    model.add(layers.Dense(512,  use_bias=False))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Dense(10, activation='softmax'))

    # / 99.46
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(layers.Flatten())
    # # Fully connected layer
    # model.add(layers.normalization.BatchNormalization())
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.normalization.BatchNormalization())
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(10, activation='softmax'))


    #99.5 /99.36
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3),
    #                         activation='relu',
    #                         input_shape=(28, 28, 1),
    #                         kernel_initializer = 'he_uniform',
    #                         kernel_regularizer = l2(0.0001)
    #                         ))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.normalization.BatchNormalization(axis=-1))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Conv2D(64, (5, 5),
    #                         activation='relu',
    #                         input_shape=(28, 28, 1),
    #                         kernel_initializer = 'he_uniform',
    #                         kernel_regularizer = l2(0.0001)))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.normalization.BatchNormalization(axis=-1))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1024, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.normalization.BatchNormalization(axis=-1))
    # model.add(layers.Dense(10, activation='softmax'))


    learning_rate_reduction =  ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.25,
                                                min_lr=0.00001)

    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])


    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=3,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images)

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(train_data, train_label, batch_size=16)
    test_generator = test_datagen.flow(val_data, val_label, batch_size=16)

    start_time = time.time()
    history = model.fit_generator(train_generator, steps_per_epoch=60000 // 16, epochs=30,
                          validation_data=test_generator, validation_steps=6000 // 16,
                                  callbacks=[learning_rate_reduction])
    training_time = time.time() - start_time
    print("Training time", training_time)

    predict_labels = model.predict_classes(predict_images)

    # merge together
    idLabel = pd.Series(data=np.arange(1, predict_labels.size + 1), name='ImageId')
    serieLabel = pd.Series(data=predict_labels, name='Label')

    pr_df = pd.concat([idLabel, serieLabel], axis=1)
    path_for_prediction = os.path.join(ROOT_PATH, 'prediction.csv')
    result = pr_df.to_csv(path_for_prediction, index=False)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


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