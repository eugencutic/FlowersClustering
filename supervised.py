from load_data import *

import numpy as np
import pandas as pd
import cv2

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras import layers
from keras import activations
from keras import Sequential
from keras.utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

from matplotlib import pyplot as plt


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


data = pd.read_csv('./data/extracted_features_ext_unscaled.csv')

file_names = []
labels = []
images_data = []

for index, row in data.iterrows():
    file_names.append(row.file)
    labels.append(row.label)
    images_data.append(row[2:].values.astype(np.float))
    
file_names = np.asarray(file_names)
labels = np.asarray(labels)
images_data = np.asarray(images_data)

# split data
sss = StratifiedShuffleSplit(test_size=0.2)
train_index, test_index = list(sss.split(images_data, labels))[0]
train_images = images_data[train_index]
test_images = images_data[test_index]
train_labels = labels[train_index]
test_labels = labels[test_index]

model = Sequential([
    layers.Dense(256, activation=activations.relu),
    layers.Dense(128, activation=activations.relu),
    layers.Dense(10, activation=activations.softmax)
])

model.compile(optimizer=Adam(learning_rate=0.01), loss=CategoricalCrossentropy())

history = model.fit(train_images, to_categorical(train_labels), epochs=10, validation_split=0.2)
plot_history(history)
predictions = model.predict_classes(test_images)

print(accuracy_score(test_labels, predictions))

# dummy = DummyClassifier(strategy='uniform')
# dummy.fit(train_images, train_labels)
# predictions = dummy.predict(test_images)
# print(accuracy_score(test_labels, predictions))