import os
import os.path as osp
import numpy as np
import shutil
import pandas as pd


ckpt_dir = f'./log/'
seqs = ['FS', 'T1', 'T2']
dicts = []
exl_path = osp.join(ckpt_dir, 'best_val.xlsx')
exl_writer = pd.ExcelWriter(exl_path)
for seq in seqs:
    seq_dir = osp.join(ckpt_dir, seq)
    best_dir = osp.join(seq_dir, 'best_models')
    if osp.exists(best_dir):
        shutil.rmtree(best_dir)
    os.makedirs(best_dir, exist_ok=False)
    all_cv_metrics = []
    for k in range(1, 6):
        cv_dir = osp.join(seq_dir, f'cross_val{k}')
        with open(osp.join(cv_dir, 'losses.txt')) as f:
            lines = f.readlines()
        lines = [i.split('_')[:-1] for i in lines]
        a = np.array(lines).astype(np.float64)
        best_epoch = int(np.where(a[:, 3] == a[:, 3].max())[0][0]) # 3 is auc, choose the best epoch by auc
        all_cv_metrics.append(a[best_epoch])
        best = f'{str(best_epoch).zfill(4)}.pth'
        shutil.copy(osp.join(cv_dir, best), osp.join(best_dir, f'{str(best_epoch).zfill(4)}val{k}.pth'))
        # print(k)
    all_cv_metrics = np.array(all_cv_metrics)
    best_fold = np.argmax(np.array(all_cv_metrics), axis=0)[3]
    for i in range(5):
        print(f'cv{all_cv_metrics[i, 0]}:ACC:{all_cv_metrics[i, 0+2]} AUC:{all_cv_metrics[i, 1+2]} F1:{all_cv_metrics[i, 2+2]} Precision:{all_cv_metrics[i, 4+2]} Recall:{all_cv_metrics[i, 3+2]}')
    _, _, ACC_mean, AUC_mean, F1_mean, recall_mean, precision_mean = all_cv_metrics.mean(axis=0)[:7].round(3)
    _, _, ACC_SD, AUC_SD, F1_SD, recall_SD, precision_SD = all_cv_metrics.std(axis=0)[:7].round(3)
    print(f'{seq}:'     
          f'ACC:{ACC_mean}±{ACC_SD} '
          f'AUC:{AUC_mean}±{AUC_SD} '
          f'F1:{F1_mean}±{F1_SD} '
          f'Precision{precision_mean}±{precision_SD}'
          f'Recall:{recall_mean}±{recall_SD} ')
    dicts.append({
        'seq': seq,
        'ACC': f'{ACC_mean}±{ACC_SD}',
        'AUC': f'{AUC_mean}±{AUC_SD}',
        'F1': f'{F1_mean}±{F1_SD}',
        'Precision': f'{precision_mean}±{precision_SD}',
        'Recall': f'{recall_mean}±{recall_SD}',
        'best_fold':best_fold+1
    })

pd.DataFrame(dicts).to_excel(excel_writer=exl_writer)
exl_writer._save()
