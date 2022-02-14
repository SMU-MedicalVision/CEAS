import shutil
import os
import os.path as osp
import numpy as np
import shutil

health = False
if health:
    exp = 'exp1'
else:
    exp = 'exp2'

ckpt_dir=f'./log_logistic_{exp}/'



best_dir = osp.join(ckpt_dir, 'best_models')
if osp.exists(best_dir):
    shutil.rmtree(best_dir)
os.makedirs(best_dir, exist_ok=False)
all_cv_metrics = []
for k in range(1, 6):
    cv_dir = osp.join(ckpt_dir, f'cross_val{k}')
    with open(osp.join(cv_dir, 'losses.txt')) as f:
        lines = f.readlines()
    lines = [i.split('_') for i in lines]
    a = np.array(lines).astype(np.float)
    best_epoch = int(np.where(a[:, 3] == a[:, 3].max())[0][0])
    all_cv_metrics.append(a[best_epoch])
    best = f'{str(best_epoch).zfill(4)}.pth'
    shutil.copy(osp.join(cv_dir, best), osp.join(best_dir, f'{str(best_epoch).zfill(4)}val{k}.pth'))
    print(k)
all_cv_metrics = np.array(all_cv_metrics)
_, _, ACC_mean, AUC_mean, F1_mean, recall_mean, precision_mean = all_cv_metrics.mean(axis=0).round(4)
_, _, ACC_SD  , AUC_SD  , F1_SD  , recall_SD  , precision_SD   = all_cv_metrics.std (axis=0).round(4)
print(f'ACC:{ACC_mean}±{ACC_SD} '
      f'AUC:{AUC_mean}±{AUC_SD} '
      f'F1:{F1_mean}±{F1_SD} '
      f'Recall:{recall_mean}±{recall_SD} '
      f'Precision{precision_mean}±{precision_SD}')
