import os
import pandas as pd
import numpy as np
import cv2


def load_images(dataset='extended'):
    if dataset == 'extended':
        images_path='./data/extended_images'
        labels_path='./data/labels/extended_labels.csv'
    if dataset == 'flower':
        images_path='./data/flower_images'
        labels_path='./data/labels/flower_labels.csv'
    if dataset == 'all':
        images_path='./data/all_images'
        labels_path='./data/labels/all_labels.csv'
    df = pd.read_csv(labels_path)
    file_names = df['file'].values.astype(str)
    labels = df['label'].values.astype(int)
    
    images = []
    for name in file_names:
        img = cv2.imread(os.path.join(images_path, name), cv2.IMREAD_COLOR)
        if img.shape != (128, 128, 3):
            img = cv2.resize(img, (128, 128))
        images.append(img)
        
    return np.array(file_names), np.array(images), np.array(labels)

