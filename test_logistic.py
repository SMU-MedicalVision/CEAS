import os
import os.path as osp
import torch
from dataset import AS_dataset_for_test, AS_dataset_logistic
from model import LR
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm

health = False
if health:
    exp = 'exp1'
else:
    exp = 'exp2'
root_dir = f'../AS_Dataset/npy_{exp}_thin'
ckpt_dir = f'./log_logistic_{exp}/best_models'
npy_dir = f'./npy/{exp}_logistic/'

os.makedirs(npy_dir)

print(ckpt_dir)
print(f'{exp}: Classification based on logistic regression')

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
dataset = AS_dataset_logistic(root=root_dir, train=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

nets = []

for k in range(0, 5):
    ckpt = os.listdir(ckpt_dir)[k]
    net = LR().cuda()
    net.load_state_dict(torch.load(osp.join(ckpt_dir, ckpt)), strict=False)
    net.eval()
    print(f'weight: {ckpt} loaded')
    nets.append(net)
dataset.iteration(1)

with torch.no_grad():
    val_acc_list = []
    y_true = torch.tensor([]).cuda()
    y_pred = torch.tensor([]).cuda()
    for i, data in tqdm(enumerate(dataloader)):
        label = data['label'].cuda()
        y = torch.zeros_like(label)
        x = data['patch'].cuda()
        for net in nets:
            y_tmp = net(x)
            y += y_tmp
        y = y / 5
        y_true = torch.cat([y_true, label.detach()])
        y_pred = torch.cat([y_pred, y.detach()])
    np.save(osp.join(npy_dir, 'y_true.npy'), np.array(y_true.cpu().squeeze(1)))
    np.save(osp.join(npy_dir, 'Ensemble_pred.npy'), np.array(y_pred.cpu().squeeze(1)))
