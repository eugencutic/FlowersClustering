import numpy as np
import pandas as pd 

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

import os

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

num_classes = 10
n_clusters_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
inertias = []
for n_clusters in n_clusters_values:
    kmeans = KMeans(n_clusters=n_clusters, max_iter=2000, n_jobs=-1)
    kmeans.fit(images_data)
    inertias.append(kmeans.inertia_)

epsilons = [1000, 1010, 1020, 1030, 1040, 1050]
clusters_count = []
for eps in epsilons:
    dbscan = DBSCAN(eps=eps, min_samples=1)
    dbscan.fit(images_data)
    clusters_count.append(len(set(dbscan.labels_)))

plt.xlabel('number of clsuters')
plt.ylabel('inertia')
plt.plot(n_clusters_values, inertias)
plt.show()

plt.xlabel('epsilon')
plt.ylabel('number of clsuters')
plt.plot(epsilons, clusters_count)
plt.show()

# fit with chosen parameters
kmeans = KMeans(n_clusters=10)
kmeans.fit(images_data)

# eps=1020 and min_samples=1 selected for 11 clusters after grid search
dbscan = DBSCAN(eps=1050, min_samples=1)
dbscan.fit(images_data)

spectral = SpectralClustering(n_clusters=10)
spectral.fit(images_data)


def save_results(model):
    groups = dict([])
    for i, label in enumerate(model.labels_):
        if label in groups:
            groups[label].append(file_names[i])
        else:
            groups[label] = [file_names[i]]

    results_df = pd.DataFrame(columns=['file', 'label'])
    i = 0
    for label in groups:
        for file_name in groups[label]:
            results_df.loc[i] = [file_name, label]
            i += 1
    
    results_df.to_csv(os.path.join('./results', model.__class__.__name__ + '_ext_unscaled.csv'), index=False)


save_results(kmeans)
save_results(dbscan)
save_results(spectral)