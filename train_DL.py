from dataset import AS_dataset
from model import resnet50
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils import *
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import argparse
from visdom import Visdom

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser('Intelligent diagnosis of Ankylosing Spondylitis(AS)')
    parser.add_argument('--k', type=int, default='5')  # 1,2,3,4,5
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--seq', type=str, default='FS')  # FS T1 T2
    parser.add_argument('--attn', type=bool, default=False)
    args = parser.parse_args()

    k = args.k
    # ----------------------VISDOM------------------------
    vis = Visdom(port=8515, env=args.seq + '_Fold' + str(k) + f"{'_attn' if args.attn else ''}")
    train_loss_vis = plot_loss(vis, 'train loss')
    val_loss_vis = plot_loss(vis, 'validation loss')
    # ----------------------INITIAL-----------------------
    root_dir = rf'../AS_Dataset/npy_thin_2024529/TAHSMURET'
    cv = f'cross_validation_2023_12_31'
    ckpt_dir = f"./log/{args.seq}/cross_val{k}"
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["CUDA_LAUNCH_BLOCKING"] = '0'
    ngpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    batch_size = 16 * ngpu
    num_workers = 2
    epochs = 200
    patch_size = [12, 256, 512]
    print(root_dir)
    print(cv)
    # ---------------------DATASET-----------------------
    train_set = AS_dataset(root=root_dir,
                           cross_validation=cv,
                           train=True,
                           patch_size=patch_size,
                           k=k, seq=args.seq, online=False)
    val_set = AS_dataset(root=root_dir,
                         cross_validation=cv,
                         train=False,
                         patch_size=patch_size,
                         k=k, seq=args.seq, online=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    print(f'Size of train_dataset:{len(train_set)}.')
    print(f'Size of val_dataset:{len(val_set)}.')
    print('Data prepared.')
    # ----------------------MODEL------------------------
    net = resnet50(in_channels=1, attention=args.attn).cuda()

    # Load model
    # ckpt_path = glob.glob(osp.join('./log_tmi_baseline', args.seq,'best_models',f'*{k}.pth'))
    # net.load_state_dict(torch.load(ckpt_path), strict=False)

    criterion = nn.BCELoss()
    if torch.cuda.is_available() and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    optimizer = optim.AdamW(net.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (0.99 ** (x - 100) if x > 100 else 1))
    # ----------------------TRAIN------------------------
    print('Start training.')
    best_loss = float('+inf')
    for epoch in tqdm(range(epochs)):
        net.train()
        train_acc_list = []
        train_loss_list = []
        y_true = torch.tensor([]).cuda()
        y_pred = torch.tensor([]).cuda()
        y_binary = torch.tensor([]).cuda()
        for i, data in enumerate(train_loader):
            # network training
            x = data['patch'].cuda()
            label = data['label'].cuda()
            net.zero_grad()
            y = net(x)
            # BCEloss weights
            weights = (712-489)*label.bool() + 489*(~label.bool())
            loss = nn.BCELoss(weight=weights)(y, label)
            loss.backward()
            # flood = torch.abs(loss-0.2)+0.2
            # flood.backward()
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

                y_true = torch.cat([y_true, label.detach()])
                y_binary = torch.cat([y_binary, (y.detach() > 0.5)])
                y_pred = torch.cat([y_pred, y.detach()])
                hit = ((y.detach() < 0.5) ^ label.bool()).sum()

                # print(torch.cat([label, y], dim=1))
                val_loss_list.append(np.array(criterion(y, label).cpu()))
                val_acc_list.append(np.array(hit.cpu()))
        val_loss = np.array(val_loss_list).mean()
        val_acc = np.array(val_acc_list).sum() / len(val_set)
        auc = roc_auc_score(y_true.cpu().squeeze(), y_pred.cpu().squeeze())

        precision, recall, F1_score, _ = precision_recall_fscore_support(y_true.int().cpu().squeeze(), y_binary.int().cpu().squeeze(),
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

        # if epoch > 100:
        save_checkpoint(net, ckpt_dir, f'{str(epoch).zfill(4)}.pth')
        if -np.array(val_acc).mean() < best_loss:
            best_loss = -np.array(val_acc).mean()
            best_epoch = epoch
            print(f'best model saved.{epoch}')
        scheduler.step()
