import cv2 
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt


def visualize_tsne_data(csv_path='./data/extracted_features_ext_unscaled.csv'):
    data_df = pd.read_csv(csv_path)

    colors = ['blue', 'orange', 'green', 'red', 'purple',
            'brown', 'pink', 'gray', 'olive', 'cyan']

    plt.figure()
    for index, row in data_df.iterrows():
        label = row.label
        data_points = row[2:].values.astype(np.float)
        plt.scatter(data_points[0], data_points[1], c=[colors[label]])

    plt.show()


def visualize_results(label=0, path='./results/KMeans_ext_unscaled.csv'):
    results_df = pd.read_csv(path)

    file_names = results_df['file'].values.astype(str)
    labels = results_df['label']

    files = [file_name for i, file_name in enumerate(file_names) if labels[i] == label]
    images = []
    for file_name in files:
        img = cv2.imread(os.path.join('./data/all_images', file_name))
        if img is None:
            print(file_name)
        if img.shape != (128, 128, 3):
            img = cv2.resize(img, (128, 128))
        images.append(img)
    images = np.asarray(images)

    n_rows, n_cols = 5, 5
    shown = False
    while (not shown):
        try:
            plt.figure(figsize=(10,10))
            for i in range(images.shape[0]):
                plt.subplot(10, 10, i+1)    
                plt.imshow(images[i])
            plt.show()
            shown = True
        except ValueError:
            n_rows += 1
            n_cols += 1

def compare_result_to_labels(result_path, labels_path = './data/labels/all_labels.csv'):
    results_df = pd.read_csv(result_path)
    labels_df = pd.read_csv(labels_path)

    results_file_names = results_df['file'].values.astype(str)
    results_labels = results_df['label']

    real_file_names = labels_df['file'].values.astype(str)
    real_labels = labels_df['label'].values

    unique_results_labels = set(results_labels)
    for result_label in unique_results_labels:
        cluster_file_names = [file_name for i, file_name in enumerate(results_file_names) 
                                        if results_labels[i] == result_label]
        real_cluster_labels = [label for i, label in enumerate(real_labels)
                                        if real_file_names[i] in cluster_file_names]
        most_common_real_label = np.argmax(np.bincount(real_cluster_labels))
        for i in range(results_labels.shape[0]):
            if results_labels[i] == result_label:
                results_labels[i] = most_common_real_label
    
    real_label_to_compare = []
    results_labels_to_compare = []

    for i in range(results_labels.shape[0]):
        file_name = results_file_names[i]
        real_label = [label for i, label in enumerate(real_labels)
                            if real_file_names[i] == file_name][0]
        result_label = results_labels[i]
        real_label_to_compare.append(real_label)
        results_labels_to_compare.append(result_label)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    print(accuracy_score(real_label_to_compare, results_labels_to_compare))
    print(classification_report(real_label_to_compare, results_labels_to_compare))
    print(confusion_matrix(real_label_to_compare, results_labels_to_compare))


visualize_tsne_data()
compare_result_to_labels('./results/SpectralClustering_ext_unscaled.csv')
visualize_results(label=0)