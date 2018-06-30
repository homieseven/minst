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

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0],28,28,1)/255.0
    x_test = x_test.reshape(x_test.shape[0],28,28,1)/255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_all = np.append(x_train, x_test, axis=0)
    y_all = np.append(y_train, y_test, axis=0)

    mean_image = np.mean(x_all, axis = 0)
    predict_images = (predict_images - mean_image)
    x_all = (x_all - mean_image)

    # split to train, val, test
    train_data = x_all[:70000]
    train_label = y_all[:70000]

    val_data = x_all[69000:70000]
    val_label = y_all[69000:70000]

    test_data = x_all[69000:70000]
    test_label = y_all[69000:70000]


    # 99.36 / 99.59 / 99.56(99.7) TOP
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    # Fully connected layer
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))


    learning_rate_reduction =  ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.25,
                                                min_lr=0.00001)

    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])


    start_time = time.time()
    history = model.fit(x=train_data, y=train_label,batch_size=16,epochs=30,validation_data=(test_data, test_label))
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