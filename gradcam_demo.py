import os
import os.path as osp
import numpy as np
import torch
from model import resnet50
from gradcam import GradCAM,GradCAMpp

from dataset import normalization_mr, random_crop, center_crop
from utils import visualize_cam
from torchvision.utils import save_image


use_CF = False
health = False
patch_size = [12, 256, 512]
if health:
    exp = 'exp1'
else:
    exp = 'exp2'

patient_paths = [f'../AS_Dataset/npy_{exp}_thin/test/AS/MR37314',
                 f'../AS_Dataset/npy_{exp}_thin/test/AS/MR101021',
                 f'../AS_Dataset/npy_{exp}_thin/test/nonAS/MR13602-A',
                 f'../AS_Dataset/npy_{exp}_thin/test/nonAS/MR70802']

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

ckpt_dir = './log_exp2/'
seqs = ['FS', 'T1', 'T2']
output_dir = './GradCAM_results/'
resnet_model_dicts = {'FS': [], 'T1': [], 'T2': []}

net = resnet50(in_channels=1, use_CF=use_CF).cuda()
net.eval()

for patient in patient_paths:
    print(patient)
    patient_dir = osp.join(output_dir, osp.basename(patient))
    for seq in seqs:
        seq_dir = osp.join(patient_dir, seq)
        ckpt_seq_dir = osp.join(ckpt_dir, seq, 'best_models')
        volume = np.load(osp.join(patient, f'{seq}.npy')).astype(np.float32)
        volume = normalization_mr(volume)
        patch = center_crop(volume, patch_size)
        patch = torch.from_numpy(patch).to(torch.float32)
        patch = patch.unsqueeze(0).unsqueeze(0).cuda()
        masks = torch.zeros_like(patch).cpu()
        _s = torch.zeros([1,1]).cpu()
        os.makedirs(seq_dir)
        for k in range(0, 5):
            ckpt = os.listdir(ckpt_seq_dir)[k]
            net.load_state_dict(torch.load(osp.join(ckpt_seq_dir, ckpt)), strict=False)
            net.eval()
            resnet_model_dict = dict(type='resnet', arch=net, layer_name='layer3', input_size=patch_size)

            resnet_gradcam = GradCAM(resnet_model_dict, False)
            mask, _ = resnet_gradcam(patch,patient)
            print(f'val{k}',_)
            mask_min, mask_max = mask.min(), mask.max()

            masks += mask.cpu()
            mask = (mask - mask_min).div(mask_max-mask_min).detach()
            patch_tmp = (patch + 1) / 2
            heatmap, result = visualize_cam(mask, patch_tmp)
            patch_tmp = patch_tmp.squeeze().unsqueeze(1).expand(12,3,256,512).cpu()
            for i, layer in enumerate(patch_tmp):

                output = torch.stack([layer, heatmap[:,i],  result[:,i]], 0)
                output_path = osp.join(seq_dir, f'{i}_val{k}.png')
                save_image(output,output_path)
            # print(_)
            _s += _.cpu()
        masks = masks / 5
        _s = _s/5
        print(_s)

        masks_min, masks_max = masks.min(), masks.max()
        print(masks_min, masks_max)
        masks = (masks - masks_min).div(masks_max-masks_min).detach()
        patch = (patch + 1) / 2
        heatmap, result = visualize_cam(masks, patch)


        patch = patch.squeeze().unsqueeze(1).expand(12,3,256,512).cpu()
        for i, layer in enumerate(patch):

            output = torch.stack([layer, heatmap[:,i],  result[:,i]], 0)
            output_path = osp.join(seq_dir, f'{i}.png')
            save_image(output,output_path)

