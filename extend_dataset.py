import os
import pandas as pd
from shutil import copy

df = pd.DataFrame(columns=['file', 'label'])
labels = []
ids = []
current_id = 211

data_path = './data/flowers'
extended_path = './data/extended'
files = os.listdir(data_path)

for file_name in files:
    label = int(file_name.split('_')[0])

    if label > 9:
        break

    labels.append(label)
    ids.append(str(current_id) + '.png')
    new_name = str(current_id) + '.png'
    current_id += 1
    
    copy(os.path.join(data_path, file_name), os.path.join(extended_path, new_name))


df['file'] = ids
df['label'] = labels

df.to_csv(os.path.join('./data', 'extended_labels.csv'), index=False)
