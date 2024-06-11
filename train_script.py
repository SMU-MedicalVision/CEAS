import os

gpus = '0'
for seq in ['T1', 'FS', 'T2']:
    for k in range(1, 6):
        # os.system('date')
        print(seq, k)
        os.system(rf'python train_DL.py --k {k} --gpus {gpus}'
                  f' --seq {seq}')
