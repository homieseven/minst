from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import sklearn
import numpy as np

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt

import os

ROOT_PATH = os.getcwd()


def main():
    predict_df = pd.read_csv(os.path.join(ROOT_PATH, 'test.csv'))
    predict_df = sklearn.utils.shuffle(predict_df)

    predict_data = predict_df.as_matrix().astype('float32') / 255
    predict_images = predict_data.reshape(predict_data.shape[0], 28, 28, 1)

    all_df = pd.read_csv(os.path.join(ROOT_PATH, 'train.csv'))
    (all_pixels_df, all_labels_df) = split_labels(all_df)

    all_pixel_data = all_pixels_df.as_matrix().astype('float32') / 255
    all_labels = all_labels_df.as_matrix()
    all_labels = to_categorical(all_labels,10)
    all_images = all_pixel_data.reshape(all_pixel_data.shape[0], 28, 28, 1)

    # split to train, val, test
    train_data = all_images[:30000]
    train_label = all_labels[:30000]

    val_data = all_images[30000:36000]
    val_label = all_labels[30000:36000]

    test_data = all_images[36000:42000]
    test_label = all_labels[36000:42000]


    optimizers = ["adam","rmsprop", "sgd", "adagrad", "adadelta", "adamax", "nadam"]
    opt_results = np.zeros((len(optimizers),2))
    iter = 4

    for op_index, opt in enumerate(optimizers):
        results = np.zeros((iter, 2))
        for i in range(0, iter):
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
            # layers.normalization.BatchNormalization(axis=-1)
            model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            # layers.normalization.BatchNormalization(axis=-1)
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            # layers.normalization.BatchNormalization(axis=-1)
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(layers.Flatten())
            # Fully connected layer

            # layers.normalization.BatchNormalization()
            model.add(layers.Dense(512, activation='relu'))
            # layers.normalization.BatchNormalization()
            # model.add(layers.Dropout(0.2))
            model.add(layers.Dense(10, activation='softmax'))

            learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                        patience=3,
                                                        verbose=1,
                                                        factor=0.5,
                                                        min_lr=0.00001)

            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(x=train_data, y=train_label, batch_size=16, epochs=1,
                                validation_data=(val_data, val_label))

            result = model.evaluate(test_data, test_label)
            print('Optimizer: {0}, iteration: {1}, results: {2}'.format(opt, i+1, result))
            array = np.array(result)
            results[i,:] = array
        #calculate average
        res = np.average(results,axis=0)
        opt_results[op_index,:] = res
    print(opt_results)

    x = np.arange(len(optimizers))

    plt.figure(1)
    plt.subplot(211)
    plt.title('Average Loss')
    plt.xticks(x, optimizers)
    plt.plot(optimizers, opt_results[:,0], 'b')

    plt.subplot(212)
    plt.title('Average accuracy')
    plt.xticks(x, optimizers)
    plt.plot(optimizers, opt_results[:,1],'r')
    plt.show()



def split_labels(data: DataFrame):
    label = data['label']
    data = data.drop(columns='label')
    return (data, label)




if __name__ == '__main__':
    main()
    #example()