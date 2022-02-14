# coding=utf-8
# loss log
from dataset import AS_dataset_logistic
from model import LR
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils import *
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import argparse

parser = argparse.ArgumentParser('Intelligent diagnosis of Ankylosing Spondylitis(AS)')
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--gpus', type=str, default='2')
parser.add_argument('--health', default=False, action='store_true')
args = parser.parse_args()
k = args.k
if args.health:
    exp = 'exp1'  # 包括health病人作为正例
else:
    exp = 'exp2'  # 只有nonAS作为正例
# ----------------------VISDOM------------------------
from visdom import Visdom

# python -m visdom.server -port=8515
vis = Visdom(port=8515, env=f'logistic_{exp}_' + str(k))
train_loss_vis = plot_loss(vis, 'train loss')
val_loss_vis = plot_loss(vis, 'validation loss')
# ----------------------INITIAL-----------------------
# _nonAS for classification of AS and non-AS
# _nonAS_health for classification of AS and non-AS&health
root_dir = f'../AS_Dataset/npy_{exp}'
cv = f'cross_validation_{exp}'
# 关闭科学计数法
torch.set_printoptions(precision=4, sci_mode=False)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
ngpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
batch_size = 200 * ngpu
num_workers = batch_size
epochs = 100

ckpt_dir = f'./log_logistic_{exp}/cross_val{k}'
os.makedirs(ckpt_dir, exist_ok=True)
# ---------------------DATASET-----------------------
print('logistic regression')
print(cv)
train_set = AS_dataset_logistic(root=root_dir, cross_validation=cv,
                                train=True,
                                k=k)
val_set = AS_dataset_logistic(root=root_dir, cross_validation=cv,
                              train=False,
                              k=k)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print(f'Size of train_dataset:{len(train_set)}.')
print(f'Size of val_dataset:{len(val_set)}.')
print('Data prepared.')
# ----------------------MODEL------------------------
net = LR().cuda()
criterion = nn.BCELoss()
if torch.cuda.is_available() and (ngpu > 1):
    net = nn.DataParallel(net, list(range(ngpu)))
    # criterion = nn.DataParallel(criterion, list(range(ngpu)))

optimizer = optim.Adam(net.parameters(), lr=5e-2)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.99 ** x)
# ----------------------TRAIN------------------------
print('Start training.')
best_loss = float('+inf')
for epoch in range(epochs):

    train_set.iteration(epoch)
    val_set.iteration(epoch)
    # -------------------training-------------------
    net.train()
    train_acc_list = []
    train_loss_list = []
    # train_f1_list = []
    # train_fpr_list = []
    # train_tpr_list = []
    y_true = torch.tensor([]).cuda()
    y_pred = torch.tensor([]).cuda()
    y_binary = torch.tensor([]).cuda()
    for i, data in tqdm(enumerate(train_loader)):
        # network training
        x = data['patch'].cuda()
        label = data['label'].cuda()

        net.zero_grad()
        y = net(x)
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()
        # losses logging
        y_binary = torch.cat([y_binary, (y.detach() > 0.5)])
        y_true = torch.cat([y_true, label.detach()])
        y_pred = torch.cat([y_pred, y.detach()])

        hit = ((y.detach() < 0.5) ^ label.bool()).sum()
        train_acc_list.append(np.array(hit.cpu()))
        train_loss_list.append(np.array(loss.detach().cpu()))

    train_loss = np.array(train_loss_list).mean()
    train_acc = np.array(train_acc_list).sum() / len(train_set)
    auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
    precision, recall, F1_score, _ = precision_recall_fscore_support(y_true.int().cpu(), y_binary.int().cpu(),
                                                                     average='binary')
    current_losses = OrderedDict()
    current_losses['BCELoss'] = train_loss
    current_losses['ACC'] = train_acc
    current_losses['AUC'] = auc
    current_losses['F1 score'] = F1_score
    current_losses['recall'] = recall
    current_losses['precision'] = precision
    train_loss_vis.plot_losses(epoch, current_losses)
    # ------------------validation------------------
    net.eval()
    val_loss_list = []
    val_acc_list = []
    y_true = torch.tensor([]).cuda()
    y_pred = torch.tensor([]).cuda()
    y_binary = torch.tensor([]).cuda()
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            x = data['patch'].cuda()
            label = data['label'].cuda()
            y = net(x)

            y_binary = torch.cat([y_binary, (y.detach() > 0.5)])
            y_true = torch.cat([y_true, label.detach()])
            y_pred = torch.cat([y_pred, y.detach()])
            hit = ((y.detach() < 0.5) ^ label.bool()).sum()

            print(torch.cat([label, y], dim=1))
            val_loss_list.append(np.array(criterion(y, label).cpu()))
            val_acc_list.append(np.array(hit.cpu()))
    val_loss = np.array(val_loss_list).mean()
    val_acc = np.array(val_acc_list).sum() / len(val_set)
    auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
    precision, recall, F1_score, _ = precision_recall_fscore_support(y_true.int().cpu(), y_binary.int().cpu(),
                                                                     average='binary')
    current_losses = OrderedDict()
    current_losses['BCELoss'] = val_loss
    current_losses['ACC'] = val_acc
    current_losses['AUC'] = auc
    current_losses['F1 score'] = F1_score
    current_losses['recall'] = recall
    current_losses['precision'] = precision
    val_loss_vis.plot_losses(epoch, current_losses)
    val_loss_log(epoch, current_losses, ckpt_dir)

    save_checkpoint(net, ckpt_dir, f'{str(epoch).zfill(4)}.pth')
    if -np.array(val_acc).mean() < best_loss:
        best_loss = -np.array(val_acc).mean()
        best_epoch = epoch
        print(f'best model saved.{epoch}')
    scheduler.step()
