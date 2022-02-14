# 医生的诊断性能
import os
import os.path as osp
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

def calculate_fpr_tpr_tnr_f1score_accuracy(y_true, y_pred):
    Tp = 0
    Fp = 0
    Tn = 0
    Fn = 0
    for label, pred in zip(y_true, y_pred):
        if (label == 1) and (pred == 1):
            Tp = Tp + 1
        elif (label == 0) and (pred == 1):
            Fp = Fp + 1
        elif (label == 0) and (pred == 0):
            Tn = Tn + 1
        elif (label == 1) and (pred == 0):
            Fn = Fn + 1
        else:
            print('something weird with labels')
            return -1
            # sys.exit()
    # calculate precision, recall, accuracy, f1
    # it's possible for division by zero in some of these cases, so do a try/except
    try:
        precision = Tp / (Tp + Fp)
    except:
        precision = 0
    try:
        recall = Tp / (Tp + Fn)
    except:
        recall = 0
    try:
        accuracy = (Tn + Tp) / (Tn + Tp + Fn + Fp)
    except:
        accuracy = 0
    try:
        f1Score = 2 * precision * recall / (precision + recall)
    except:
        f1Score = 0
    try:
        fpr = Fp / (Fp + Tn)
    except:
        fpr = 0
    try:
        tpr = Tp / (Tp + Fn)
    except:
        tpr = 0
    try:
        tnr = Tn / (Tn + Fp)
    except:
        tnr = 0
    return (fpr, tpr, tnr, f1Score, accuracy)

exp = 'exp1'
root_dir = r'../AS_Dataset'
exl = pd.read_excel('clinical_info.xlsx', sheet_name=None, index_col=0,
                    header=0)

AS_patient = os.listdir(osp.join(root_dir, f'npy_{exp}_thin', 'test', 'AS'))
AS_patient = list(map(lambda i: i.split('-')[0], AS_patient))
nonAS_patient = os.listdir(osp.join(root_dir, f'npy_{exp}_thin', 'test', 'nonAS'))
nonAS_patient = list(map(lambda i: i.split('-')[0], nonAS_patient))
label = []
yq_pred = []
cyj_pred = []
print('AS')
print(len(AS_patient),len(nonAS_patient))
for patient in AS_patient:
    try:
        print(patient,exl['AS'].loc[patient]['YQ(AS:2,nonAS:1,healthy:0)'],exl['AS'].loc[patient]['CYJ(AS:2,nonAS:1,healthy:0)'])
        if exl['AS'].loc[patient]['YQ(AS:2,nonAS:1,healthy:0)'] == 2:
            yq_pred.append(1)
        else:
            yq_pred.append(0)
        if exl['AS'].loc[patient]['CYJ(AS:2,nonAS:1,healthy:0)'] == 2:
            cyj_pred.append(1)
        else:
            cyj_pred.append(0)
        label.append(1)
    except KeyError:
        print(patient, 'not found in excel')
        continue
print('nonAS')
for patient in nonAS_patient:
    try:

        if patient in exl['non-AS'].index:
            print(patient, exl['non-AS'].loc[patient]['YQ(AS:2,nonAS:1,healthy:0)'],
                  exl['non-AS'].loc[patient]['CYJ(AS:2,nonAS:1,healthy:0)'])
            pred = exl['non-AS'].loc[patient]['YQ(AS:2,nonAS:1,healthy:0)']
            if not isinstance(pred, np.int64):
                pred = pred.values[0]
            if pred != 2:
                yq_pred.append(0)
            else:
                yq_pred.append(1)
            pred = exl['non-AS'].loc[patient]['CYJ(AS:2,nonAS:1,healthy:0)']
            if not isinstance(pred, np.int64):
                pred = pred.values[0]
            if pred != 2:
                cyj_pred.append(0)
            else:
                cyj_pred.append(1)
            label.append(0)
        elif patient in exl['health'].index:
            print(patient, exl['health'].loc[patient]['YQ(AS:2,nonAS:1,healthy:0)'],
                  exl['health'].loc[patient]['CYJ(AS:2,nonAS:1,healthy:0)'])
            pred = exl['health'].loc[patient]['YQ(AS:2,nonAS:1,healthy:0)']
            if not isinstance(pred, np.int64):
                pred = pred.values[0]
            if pred != 2:
                yq_pred.append(0)
            else:
                yq_pred.append(1)
            pred = exl['health'].loc[patient]['CYJ(AS:2,nonAS:1,healthy:0)']
            if not isinstance(pred, np.int64):
                pred = pred.values[0]
            if pred != 2:
                cyj_pred.append(0)
            else:
                cyj_pred.append(1)
            label.append(0)
        else:
            print('!!!!!!!!!error')
    except KeyError:
        print(patient, 'not found in excel')
        continue
print(len(label))
np.save(osp.join('npy_for_roc',exp,'radiologist1.npy'),np.array(yq_pred))
np.save(osp.join('npy_for_roc',exp,'radiologist2.npy'),np.array(cyj_pred))
