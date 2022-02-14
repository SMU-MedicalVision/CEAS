import os
import os.path as osp
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
exp = 'exp2'
root_dir = rf'../AS_Dataset/npy_{exp}_thin/train'
target_list = []
data_list = []
test_size = 0.2
skf = StratifiedKFold(n_splits=5)
os.makedirs(f'cross_validation_{exp}')
for c in os.listdir(root_dir):
    if c == 'AS':
        label = 1
    else:
        label = 0
    for patient_dir in os.listdir(osp.join(root_dir, c)):
        data_list.append(patient_dir)
        target_list.append(label)
data_list = np.array(data_list)
target_list = np.array(target_list)
for k, (train_index, test_index) in enumerate(skf.split(data_list, target_list)):
    np.save(osp.join(f'cross_validation_{exp}', f'new_val{k + 1}'), np.array(data_list[test_index]))
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data_list[train_index], data_list[test_index]
    y_train, y_test = target_list[train_index], target_list[test_index]
