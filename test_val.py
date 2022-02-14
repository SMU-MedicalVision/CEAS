import glob
import os
import os.path as osp

import pandas as pd
import torch
from dataset import AS_dataset_for_test_val
from model import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, cohen_kappa_score, \
    confusion_matrix
import scikits.bootstrap as boot
def stat_func(a):
    y = a[:, 0]
    pred = a[:, 1]
    binary_pred = pred > 0.5
    # sensitive & spec
    [[TN, FP], [FN, TP]] = confusion_matrix(y, binary_pred)
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    acc = accuracy_score(y, binary_pred)
    auc = roc_auc_score(y, pred)
    precision, recall, F1_score, _ = precision_recall_fscore_support(y, binary_pred, average='binary')
    kappa = cohen_kappa_score(y, binary_pred)
    return auc, acc, F1_score, precision, recall, sens, spec, kappa


def stat_func_radiologist(a):
    y = a[:, 0]
    binary_pred = a[:, 1]
    # binary_pred = pred > 0.5
    # sensitive & spec
    [[TN, FP], [FN, TP]] = confusion_matrix(y, binary_pred)
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    acc = accuracy_score(y, binary_pred)
    precision, recall, F1_score, _ = precision_recall_fscore_support(y, binary_pred, average='binary')
    kappa = cohen_kappa_score(y, binary_pred)
    return acc, F1_score, precision, recall, sens, spec, kappa


def keys_count(y, pred):
    binary_pred = pred > 0.5
    [[TN, FP], [FN, TP]] = confusion_matrix(y, binary_pred)
    count_dict = {'ACC': [TN + TP, TN + FP + FN + TP],
                  'Precision': [TP, TP + FP],
                  'Recall': [TP, TP + FN],
                  'Sensitivity': [TP, TP + FN],
                  'Specificity': [TN, TN + FP], }
    return count_dict


def CI_Calc(y, pred, radiologist=False):
    index_dict = {}
    keys = ['AUROC', 'ACC', 'F1 Score', 'Precision', 'Recall', 'Sensitivity', 'Specificity', "Cohen's κ Score"]
    keys_to_be_counted = ['ACC', 'Precision', 'Recall', 'Sensitivity', 'Specificity']
    count = keys_count(y, pred)
    if radiologist:
        ci_lower, ci_upper = boot.ci(np.stack([y, pred], axis=1), stat_func_radiologist, seed=42).round(3)
        mean = ((ci_lower + ci_upper) / 2).round(3)
        ci_lower = np.concatenate([['-'], ci_lower])
        ci_upper = np.concatenate([['-'], ci_upper])
        mean = np.concatenate([['-'], mean])


    else:
        ci_lower, ci_upper = boot.ci(np.stack([y, pred], axis=1), stat_func, seed=42).round(3)
        mean = ((ci_lower + ci_upper) / 2).round(3)

    for i, key in enumerate(keys):
        if key in keys_to_be_counted:
            index_dict.update({key: f'{mean[i]}({count[key][0]}/{count[key][1]})\n[{ci_lower[i]},{ci_upper[i]}]'})
        else:
            index_dict.update({key: f'{mean[i]}\n[{ci_lower[i]},{ci_upper[i]}]'})
    return index_dict
os.system('date')

use_CF = True
health = True
if health:
    exp = 'exp1'
else:
    exp = 'exp2'
root_dir = f'../AS_Dataset/npy_{exp}_thin'
cv = f'cross_validation_{exp}'
if use_CF:
    ckpt_dir = f'./log_{exp}_HLA/'
    npy_dir = f'./npy/{exp}_HLA/'
else:
    ckpt_dir = f'./log_{exp}/'
    npy_dir = f'./npy/{exp}/'
# ckpt_dir='log_exp1_HLA_old_version_no_ESR_CRP'
# TODO 验证去掉benchmark加速后的结果有无差异
seqs = ['FS', 'T1', 'T2']
patch_size = [12, 256, 512]
print(root_dir)
print(cv)
print(ckpt_dir)
print(f'using additional HLA-B27 info? {use_HLA}')
print(f'{exp}: Classification based on FS, T1 and T2 sequence')

# torch.set_printoptions(precision=4, sci_mode=False)
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
cv = f'cross_validation_{exp}'
all_dicts = []
separated_dicts = {'Multi-modality': {'y': [], 'pred': []},
                   'FS': {'y': [], 'pred': []},
                   'T1': {'y': [], 'pred': []},
                   'T2': {'y': [], 'pred': []}}
for k in tqdm(range(1, 6)):
    print(k)
    dataset = AS_dataset_for_test_val(root=root_dir,
                                      cross_validation=cv,
                                      patch_size=patch_size,
                                      k=k, use_HLA=use_HLA)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    nets = {}
    for seq in seqs:
        seq_dir = osp.join(ckpt_dir, seq, 'best_models')
        ckpt = glob.glob(osp.join(seq_dir, f'*val{k}.pth'))[0]
        net = resnet50(in_channels=1, use_hla=use_HLA).cuda()
        net.load_state_dict(torch.load(ckpt), strict=False)
        net.eval()
        print(f'weight: {ckpt} loaded')
        nets.update({seq: net})
    if use_HLA:
        dataset.iteration(0)
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
        for i, data in enumerate(dataloader):
            patients.append(data['patient'][0])
            label = data['label'].cuda()
            y = torch.zeros_like(label)
            # flatten = torch.zeros_like(label).cuda()
            for seq in seqs:
                if use_HLA:
                    x = [data['patch'][seq].cuda(), data['hla'].cuda()]
                else:
                    x = data['patch'][seq].cuda()

                y_tmp = nets[seq](x)
                y += y_tmp
                separated_y = y_tmp

                separated_pred[seq] = torch.cat([separated_pred[seq], separated_y.detach()])
                separated_binary[seq] = torch.cat([separated_binary[seq], (separated_y.detach() > 0.5)])
                separated_hit = ((separated_y.detach() < 0.5) ^ label.bool()).sum()
                separated_val_acc_list[seq].append(np.array(separated_hit.cpu()))
            y = y / 3

            y_true = torch.cat([y_true, label.detach()])
            y_pred = torch.cat([y_pred, y.detach()])
            y_binary = torch.cat([y_binary, (y.detach() > 0.5)])
            hit = ((y.detach() < 0.5) ^ label.bool()).sum()

            # print(torch.cat([label, y], dim=1))
            val_acc_list.append(np.array(hit.cpu()))
        y = list(y_true.int().cpu())
        pred = list(y_pred.cpu())
        separated_dicts['Multi-modality']['y'] += y
        separated_dicts['Multi-modality']['pred'] += pred
        binary_pred = y_binary.int().cpu()

        # [[TN, FP], [FN, TP]] = confusion_matrix(y, binary_pred)
        # sens = TP / (TP + FN)
        # spec = TN / (TN + FP)
        # val_acc = np.array(val_acc_list).sum() / len(dataset)
        # auc = roc_auc_score(y, pred).round(3)
        # precision, recall, F1_score, _ = precision_recall_fscore_support(y, binary_pred,
        #                                                                  average='binary')
        # kappa = cohen_kappa_score(y, binary_pred)
        #
        # mean = [auc.round(3),
        #         val_acc.round(3),
        #         F1_score.round(3),
        #         precision.round(3),
        #         recall.round(3),
        #         sens.round(3),
        #         spec.round(3),
        #         kappa.round(3)]
        #
        # # print(f'ACC:{val_acc.round(4)}  AUC:{auc}  F1:{F1_score.round(4)} '
        # #       f'Precision:{precision.round(4)}  Recall:{recall.round(4)}')
        # keys = ['AUROC', 'ACC', 'F1 Score', 'Precision', 'Recall', 'Sensitivity', 'Specificity', "Cohen's κ Score"]
        # keys_to_be_counted = ['ACC', 'Precision', 'Recall', 'Sensitivity', 'Specificity']
        # ensemble_dict = {}
        # for i, key in enumerate(keys):
        #     if key in keys_to_be_counted:
        #         ensemble_dict.update({key: f'{mean[i]}({count[key][0]}/{count[key][1]})[{ci_lower[i]},{ci_upper[i]}]'})
        #     else:
        #         ensemble_dict.update({key: f'{mean[i]}[{ci_lower[i]},{ci_upper[i]}]'})
        # ensemble_dict.update({'model': 'Multi-modality'})

    print('-------------------Separated Classification-------------------')
    # separated_dicts = []
    for seq in seqs:
        y = y_true.int().cpu()
        pred = separated_pred[seq].cpu()
        separated_dicts[seq]['y'] += y
        separated_dicts[seq]['pred'] += pred
        # binary_pred = pred > 0.5
        # [[TN, FP], [FN, TP]] = confusion_matrix(y, binary_pred)
        # sens = TP / (TP + FN)
        # spec = TN / (TN + FP)
        val_acc = accuracy_score(y,pred>0.5).round(3)#np.array(val_acc_list).sum() / len(dataset)
        auc = roc_auc_score(y, pred).round(3)
        print(seq,auc,val_acc)
        # precision, recall, F1_score, _ = precision_recall_fscore_support(y, binary_pred,
        #                                                                  average='binary')
        # kappa = cohen_kappa_score(y, binary_pred)
        #
        # mean = [auc.round(3),
        #         val_acc.round(3),
        #         F1_score.round(3),
        #         precision.round(3),
        #         recall.round(3),
        #         sens.round(3),
        #         spec.round(3),
        #         kappa.round(3)]
        #
        # keys = ['AUROC', 'ACC', 'F1 Score', 'Precision', 'Recall', 'Sensitivity', 'Specificity', "Cohen's κ Score"]
        # keys_to_be_counted = ['ACC', 'Precision', 'Recall', 'Sensitivity', 'Specificity']
        # index_dict = {}
        # for i, key in enumerate(keys):
        #     if key in keys_to_be_counted:
        #         index_dict.update({key: f'{mean[i]}({count[key][0]}/{count[key][1]})[{ci_lower[i]},{ci_upper[i]}]'})
        #     else:
        #         index_dict.update({key: f'{mean[i]}[{ci_lower[i]},{ci_upper[i]}]'})
        # index_dict.update({'model': seq})
        # separated_dicts.append(index_dict)

for j, model in tqdm(enumerate(separated_dicts.keys())):
    pred = np.array(separated_dicts[model]['pred'])
    y = np.array(separated_dicts[model]['y'])
    mean_CI_dict = CI_Calc(y, pred, 'radiologist' in model)
    mean_CI_dict.update({'model': model.split('.')[0]})
    all_dicts.append(mean_CI_dict)

# log to excel
exl_path = osp.join(ckpt_dir, 'val_test_result.xlsx')
exl_writer = pd.ExcelWriter(exl_path)
pd.DataFrame(all_dicts).set_index('model').to_excel(excel_writer=exl_writer)
exl_writer.save()
os.system('date')
