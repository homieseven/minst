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

    (created_models, model_titles) = create_models()
    model_results = np.zeros((len(created_models),2))
    number_of_iterations = 1

    for m_index, model in enumerate(created_models):
        results = np.zeros((number_of_iterations, 2))
        for i in range(0, number_of_iterations):

            learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                        patience=3,
                                                        verbose=1,
                                                        factor=0.5,
                                                        min_lr=0.00001)

            model.compile(optimizer="adamax", loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(x=train_data, y=train_label, batch_size=32, epochs=1,
                                validation_data=(val_data, val_label))

            result = model.evaluate(test_data, test_label)
            print('Model: {0}, iteration: {1}, results: {2}'.format(m_index, i+1, result))
            array = np.array(result)
            results[i,:] = array
        #calculate average
        res = np.average(results,axis=0)
        model_results[m_index,:] = res
    print(model_results)

    x = np.arange(len(created_models))

    plt.figure(1)
    plt.subplot(211)
    plt.title('Average Loss')
    plt.xticks(x,model_titles)
    plt.plot(model_titles, model_results[:,0], 'b')

    plt.subplot(212)
    plt.title('Average accuracy')
    plt.xticks(x, model_titles)
    plt.plot(model_titles, model_results[:,1],'r')
    plt.show()


def split_labels(data: DataFrame):
    label = data['label']
    data = data.drop(columns='label')
    return (data, label)


def create_models():
    created_models = []
    model_titles = []


    #---------------------------------------------------------------------------------
    #MODEL 1
    #---------------------------------------------------------------------------------
    model1 = models.Sequential()
    model1.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model1.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model1.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model1.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model1.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model1.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model1.add(layers.Flatten())
    # Fully connected layer
    model1.add(layers.Dense(256, activation='relu'))
    model1.add(layers.Dense(10, activation='softmax'))
    created_models.append(model1)
    model_titles.append("Model 1")


    #---------------------------------------------------------------------------------
    #MODEL 2
    #---------------------------------------------------------------------------------
    model2 = models.Sequential()
    model2.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model2.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model2.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model2.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model2.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model2.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model2.add(layers.Flatten())
    # Fully connected layer
    model2.add(layers.Dense(128, activation='relu'))
    model2.add(layers.Dense(10, activation='softmax'))
    created_models.append(model2)
    model_titles.append("Model 2")

    # ---------------------------------------------------------------------------------
    # MODEL 3
    # ---------------------------------------------------------------------------------
    model3 = models.Sequential()
    model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model3.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model3.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model3.add(layers.Flatten())
    # Fully connected layer
    model3.add(layers.Dense(256, activation='relu'))
    model3.add(layers.Dense(10, activation='softmax'))
    created_models.append(model3)
    model_titles.append("Model 3")

    return (created_models, model_titles)



if __name__ == '__main__':
    main()
    #example()

