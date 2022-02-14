import numpy as np
import torch
import torch.nn as nn
import os
import os.path as osp
import cv2


def save_checkpoint(model, ckpt_dir, name):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(ckpt_dir, name))
    else:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, name))


class plot_loss():
    def __init__(self, vis, title, update='append'):
        self.vis = vis
        self.win = title
        self.title = title
        self.update = update

    def plot_losses(self, step, losses):
        legends = losses.keys()
        losses = ([[tensor2float(losses[k])] for k in legends])
        self.vis.line(
            X=np.array([step]),
            #   X=np.array(range(0, step + 1)),
            Y=np.array(losses),
            opts={
                'title': self.title + ' loss over time',
                'legend': list(legends),
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.win,
            update=self.update)


def tensor2float(input_error):
    if torch.is_tensor(input_error):
        error = input_error.item()
    elif isinstance(input_error, torch.autograd.Variable):
        error = input_error.data[0]
    else:
        error = input_error
    return error


def val_loss_log(epoch, current_losses, ckpt_dir):
    with open(osp.join(ckpt_dir, 'losses.txt'), 'a') as f:
        print(
            f'{epoch}_{current_losses["BCELoss"]}_{current_losses["ACC"]}_{current_losses["AUC"]}_{current_losses["F1 score"]}_{current_losses["recall"]}_{current_losses["precision"]}',
            file=f)


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def acm(x, cmap):
    y = np.zeros((*x.shape, 3))
    for i, layer in enumerate(x):
        y[i] = cv2.applyColorMap(layer, cmap)
    return y


def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = acm(np.uint8(255 * mask.squeeze().cpu()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(3, 0, 1, 2).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result
