import numpy as np
import os
import os.path as osp
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, cohen_kappa_score, \
    confusion_matrix
import scikits.bootstrap as boot
import pandas as pd
from tqdm import tqdm

exps = ['exp1', 'exp2']
# models = ['Ensemble_pred.npy',
#           'Ensemble_pred_wo_CF.npy',
#           'logistic_pred.npy',
#           'radiologist1.npy',
#           'radiologist2.npy']

models = ['T1_pred.npy',
          'T2_pred.npy',
          'FS_pred.npy',
          'Ensemble_pred.npy']

npy_dir = 'npy_for_roc'


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
            index_dict.update({key: f'{mean[i]}({count[key][0]}/{count[key][1]})[{ci_lower[i]},{ci_upper[i]}]'})
        else:
            index_dict.update({key: f'{mean[i]}[{ci_lower[i]},{ci_upper[i]}]'})
    return index_dict


for i, exp in enumerate(exps):
    all_dicts = []
    y = np.load(osp.join(npy_dir, exp, 'y_true.npy'))
    for j, model in tqdm(enumerate(models)):
        pred = np.load(osp.join(npy_dir, exp, model))

        mean_CI_dict = CI_Calc(y, pred, 'radiologist' in model)
        mean_CI_dict.update({'model': model.split('.')[0]})
        all_dicts.append(mean_CI_dict)

    exl_path = osp.join(npy_dir, exp, 'result2.xlsx')
    exl_writer = pd.ExcelWriter(exl_path)
    pd.DataFrame(all_dicts).set_index('model').to_excel(excel_writer=exl_writer)
    exl_writer.save()
