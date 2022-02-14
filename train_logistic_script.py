#
import os
import argparse
gpus = 2
# health = False
# HLA = True
for health in [True, False]:
    for k in range(1, 6):
        # os.system('which python')
        # os.system('pwd')
        os.system('date')
        os.system(f'/home/czfy/python train_logistic.py --k {k} '
                  f'--gpus {gpus}'
                  f'{" --health"if health else ""}')

