import os
import os.path as osp
from glob import glob
import pandas as pd
import torch
from dataset import AS_dataset_for_test_and_train
from model import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
root_dir = rf'../AS_Dataset/npy_thin_2024529/TAHSMURET'
ckpt_dir = f'./log/'
torch.set_printoptions(precision=4, sci_mode=False)

use_best_fold = False # True只用最好的fold，False用所有fold
npy_dir = f'npy_ResNet50{"_best_fold" if use_best_fold else ""}/TAHSMURET_test'

# TODO 验证去掉benchmark加速后的结果有无差异
seqs = ['FS', 'T1', 'T2']

print(root_dir)
print(ckpt_dir)
print(f'Classification based on FS, T1 and T2 sequence')

dataset = AS_dataset_for_test_and_train(root=root_dir, patch_size=[12, 256, 512])
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
os.makedirs(npy_dir,exist_ok=True)
nets = {'FS': [], 'T1': [], 'T2': []}
for seq in seqs:
    seq_dir = osp.join(ckpt_dir, seq, 'best_models')
    if use_best_fold:
        best_fold = pd.read_excel(osp.join(ckpt_dir, 'best_val.xlsx'),index_col=1).loc[seq]['best_fold']
        ckpt = glob(osp.join(seq_dir, '*val'+str(best_fold)+'.pth'))[0]
        net = resnet50(in_channels=1).cuda()
        net.load_state_dict(torch.load(ckpt), strict=False)
        print(f'weight: {ckpt} loaded')
        nets[seq].append(net)
    else:
        for k in range(0, 5):
            ckpt = os.listdir(seq_dir)[k]
            net = resnet50(in_channels=1).cuda()
            net.load_state_dict(torch.load(osp.join(seq_dir, ckpt)), strict=False)
            net.eval()
            print(f'weight: {ckpt} loaded')
            nets[seq].append(net)

with torch.no_grad():
    val_acc_list = []
    y_true = torch.tensor([]).cuda()
    y_pred = torch.tensor([]).cuda()
    y_binary = torch.tensor([]).cuda()
    flattens = torch.tensor([]).cuda()
    separated_pred = {'FS': torch.tensor([]).cuda(),
                      'T1': torch.tensor([]).cuda(),
                      'T2': torch.tensor([]).cuda()}
    separated_binary = {'FS': torch.tensor([]).cuda(),
                        'T1': torch.tensor([]).cuda(),
                        'T2': torch.tensor([]).cuda()}
    separated_val_acc_list = {'FS': [],
                              'T1': [],
                              'T2': []}
    patients = []
    for i, data in tqdm(enumerate(dataloader)):
        patients.append(data['patient'][0])
        label = data['label'].cuda()
        y = torch.zeros_like(label)
        for seq in seqs:

            x = data['patch'][seq].cuda()
            separated_y = torch.zeros_like(label)
            for net in nets[seq]:
                y_tmp = net(x)
                y += y_tmp
                separated_y += y_tmp
            separated_y = separated_y / len(nets[seq])
            separated_pred[seq] = torch.cat([separated_pred[seq], separated_y.detach()])
            separated_binary[seq] = torch.cat([separated_binary[seq], (separated_y.detach() > 0.5)])
            separated_hit = ((separated_y.detach() < 0.5) ^ label.bool()).sum()
            separated_val_acc_list[seq].append(np.array(separated_hit.cpu()))
        y = y / (len(nets[seq]) * len(seqs))

        y_true = torch.cat([y_true, label.detach()])
        y_pred = torch.cat([y_pred, y.detach()])
        y_binary = torch.cat([y_binary, (y.detach() > 0.5)])
        hit = ((y.detach() < 0.5) ^ label.bool()).sum()

        # print(torch.cat([label, y], dim=1))
        val_acc_list.append(np.array(hit.cpu()))

    val_acc = np.array(val_acc_list).sum() / len(dataset)
    auc = roc_auc_score(y_true.cpu(), y_pred.cpu()).round(4)

    np.save(osp.join(npy_dir, 'y_true.npy'), np.array(y_true.cpu().squeeze(1)))
    np.save(osp.join(npy_dir, 'Ensemble_pred.npy'), np.array(y_pred.cpu().squeeze(1)))
    np.save(osp.join(npy_dir, 'patients.npy'), np.array(patients))

    precision, recall, F1_score, _ = precision_recall_fscore_support(y_true.int().cpu(), y_binary.int().cpu(),
                                                                     average='binary')
    print(f'ACC:{val_acc.round(4)}  AUC:{auc}  F1:{F1_score.round(4)} '
          f'Precision:{precision.round(4)}  Recall:{recall.round(4)}')
    ensemble_dict = {
        'seq': 'Ensemble',
        'ACC': val_acc.round(3),
        'AUC': auc.round(3),
        'F1': F1_score.round(3),
        'Precision': precision.round(3),
        'Recall': recall.round(3)
    }
    diff = np.array((y_pred - y_true).cpu().squeeze())
    patient_diff = list(zip(patients, diff))
    patient_diff = sorted(patient_diff, key=lambda i: i[1])

print('-------------------Separated Classification-------------------')
separated_dicts = []
for seq in seqs:
    val_acc = np.array(separated_val_acc_list[seq]).sum() / len(dataset)
    auc = roc_auc_score(y_true.cpu(), separated_pred[seq].cpu()).round(4)
    np.save(osp.join(npy_dir, f'{seq}_pred.npy'), np.array(separated_pred[seq].cpu().squeeze(1)))
    precision, recall, F1_score, _ = precision_recall_fscore_support(y_true.int().cpu(),
                                                                     separated_binary[seq].int().cpu(),
                                                                     average='binary')
    print(f'{seq}: ACC:{val_acc.round(4)}  AUC:{auc}  F1:{F1_score.round(4)} Precision:{precision.round(4)}\
      Recall:{recall.round(4)}')
    separated_dicts.append({
        'seq': seq,
        'ACC': val_acc.round(3),
        'AUC': auc.round(3),
        'F1': F1_score.round(3),
        'Precision': precision.round(3),
        'Recall': recall.round(3)
    })
print(patient_diff)
# log to excel
all_dicts = separated_dicts + [ensemble_dict]
exl_path = osp.join(npy_dir, 'result2seqs.xlsx')
exl_writer = pd.ExcelWriter(exl_path)
pd.DataFrame(all_dicts).to_excel(excel_writer=exl_writer)
exl_writer._save()

