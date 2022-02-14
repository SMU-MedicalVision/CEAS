import os
import os.path as osp

import pandas as pd
import torch
from dataset import AS_dataset_for_test
from model import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import time
from tqdm import tqdm

use_CF = True
health = True
if health:
    exp = 'exp1'
else:
    exp = 'exp2'
root_dir = f'../AS_Dataset/npy_{exp}_thin'
cv = f'cross_validation_{exp}'
if use_CF:
    ckpt_dir = f'./log_{exp}_CF/'
    npy_dir = f'./npy/{exp}_CF/'
else:
    ckpt_dir = f'./log_{exp}/'
    npy_dir = f'./npy/{exp}/'

seqs = ['FS', 'T1', 'T2']

print(root_dir)
print(cv)
print(ckpt_dir)
print(f'using additional clinical feature info? {use_CF}')
print(f'{exp}: Classification based on FS, T1 and T2 sequence')


os.environ["CUDA_VISIBLE_DEVICES"] = '4'

dataset = AS_dataset_for_test(root=root_dir, patch_size=[12, 256, 512], use_CF=use_CF)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
os.makedirs(npy_dir)
nets = {'FS': [], 'T1': [], 'T2': []}
for seq in seqs:
    seq_dir = osp.join(ckpt_dir, seq, 'best_models')
    for k in range(0, 5):
        ckpt = os.listdir(seq_dir)[k]
        net = resnet50(in_channels=1, use_CF=use_CF).cuda()
        net.load_state_dict(torch.load(osp.join(seq_dir, ckpt)), strict=False)
        net.eval()
        print(f'weight: {ckpt} loaded')
        nets[seq].append(net)
if use_CF:
    dataset.iteration(0)
with torch.no_grad():
    val_acc_list = []
    y_true = torch.tensor([]).cuda()
    y_pred = torch.tensor([]).cuda()
    flattens = torch.tensor([]).cuda()
    separated_pred = {'FS': torch.tensor([]).cuda(),
                      'T1': torch.tensor([]).cuda(),
                      'T2': torch.tensor([]).cuda()}
    patients = []
    for i, data in tqdm(enumerate(dataloader)):
        patients.append(data['patient'][0])
        label = data['label'].cuda()
        y = torch.zeros_like(label)
        for seq in seqs:
            if use_CF:
                x = [data['patch'][seq].cuda(), data['CF'].cuda()]
            else:
                x = data['patch'][seq].cuda()
            separated_y = torch.zeros_like(label)
            for net in nets[seq]:
                y_tmp = net(x)
                y += y_tmp
                separated_y += y_tmp
            separated_y = separated_y / 5
            separated_pred[seq] = torch.cat([separated_pred[seq], separated_y.detach()])
        y = y / (5 * len(seqs))
        y_true = torch.cat([y_true, label.detach()])
        y_pred = torch.cat([y_pred, y.detach()])

    np.save(osp.join(npy_dir, 'y_true.npy'), np.array(y_true.cpu().squeeze(1)))
    np.save(osp.join(npy_dir, 'Ensemble_pred.npy'), np.array(y_pred.cpu().squeeze(1)))

    diff = np.array((y_pred - y_true).cpu().squeeze())
    patient_diff = list(zip(patients, diff))
    patient_diff = sorted(patient_diff, key=lambda i: i[1])
    print(patient_diff)
# -------------------Separated Classification-------------------
separated_dicts = []
for seq in seqs:
    auc = roc_auc_score(y_true.cpu(), separated_pred[seq].cpu()).round(4)
    np.save(osp.join(npy_dir, f'{seq}_pred.npy'), np.array(separated_pred[seq].cpu().squeeze(1)))

