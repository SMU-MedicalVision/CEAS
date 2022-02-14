#
import os
import argparse
parser = argparse.ArgumentParser('Intelligent diagnosis of Ankylosing Spondylitis(AS)')
parser.add_argument('--use_CF', default=True, action='store_true')
parser.add_argument('--health', default=True, action='store_true')
parser.add_argument('--gpus', type=str, default='0')
args = parser.parse_args()
print(args.gpus)
for k in range(1, 6):
    # os.system('which python')
    # os.system('pwd')
    os.system('date')
    os.system(f'/home/czfy/python train.py --k {k} --gpus {args.gpus}'
              f'{" --health"if args.health else ""}'
              f'{" --use_CF"if args.use_CF else ""}'
              f' --seq FS')
    print(k)
