from load_data import *

import numpy as np
import pandas as pd
import cv2

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

file_names, images, labels = load_images(dataset='flower')

input_images = preprocess_input(images)
model = VGG16(include_top=False, input_shape=(128, 128, 3))

features = model.predict(input_images)
features = features.reshape(features.shape[0], -1)

# bring data between 0 and 1
# scaler = MinMaxScaler()
# features_scaled = scaler.fit_transform(features)

# pca for choosing components which explain at least 95% variance
pca = PCA(n_components=.95)
features_reduced_pca = pca.fit_transform(features)

# tsne for data visualization
tsne = TSNE(perplexity=50, n_iter=2000, early_exaggeration=12, n_iter_without_progress=1000)
features_reduced_tsne = tsne.fit_transform(features)


def save_features(features, path):
    extracted_feature_df = pd.DataFrame()
    extracted_feature_df['file'] = file_names
    extracted_feature_df['label'] = labels
    for i in range(features.shape[1]):
        extracted_feature_df['feature_' + str(i)] = features[:, i]

    extracted_feature_df.to_csv(path, index=False)


save_features(features, './data/vgg_features.csv')