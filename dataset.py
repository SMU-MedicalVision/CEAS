import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import pandas as pd
import warnings
from monai.transforms import (Compose,
                              ToTensor,
                              RandFlip,
                              RandGaussianSmooth,
                              RandHistogramShift,
                              RandGibbsNoise,
                              RandAdjustContrast,
                              RandBiasField,
                              ScaleIntensityRangePercentiles,
                              EnsureChannelFirst,
                              RandGaussianNoise,
                              RandSpatialCrop,
                              CenterSpatialCrop)

warnings.filterwarnings('ignore')


def normalization_mr(X):
    p2, p99 = np.percentile(X, (2, 99))
    # result = np.clip(X,p2,p99)
    result = ((X - p2) / (p99 - p2)).astype('float')
    # result = (result - 0.5) * 2
    # result[result > 1] = 1
    # result[result < 0] = 0
    return result * 2 - 1


def normalization_mr_vit(X):
    p2, p99 = np.percentile(X, (2, 99))
    # result = np.clip(X,p2,p99)
    result = ((X - p2) / (p99 - p2)).astype('float')
    # result = (result - 0.5) * 2
    result[result > 1] = 1
    result[result < 0] = 0
    return result


def random_crop_with_mask(volume, mask_point, mask_size, patch_size, patient):
    flag = mask_size > patch_size
    new_mask_size = patch_size * flag + mask_size * ~flag
    differential = new_mask_size - mask_size
    mask_point -= (differential / 2).round().astype(int)
    image_size = np.array(volume.shape)
    # patch_size - mask_size
    p1 = np.max([[0., 0., 0.], mask_point + new_mask_size - patch_size], axis=0)
    p2 = np.min([mask_point, image_size - patch_size], axis=0)
    z, x, y = np.random.uniform(p1, p2).round().astype(int)
    patch = volume[z:z + patch_size[0], x:x + patch_size[1], y:y + patch_size[2]]
    assert patch.shape[0] == 12, (patient, patch.shape)
    assert patch.shape[1] == 256, (patient, patch.shape)
    assert patch.shape[2] == 512, (patient, patch.shape)
    return patch


def random_crop(volume, patch_size):
    p1 = np.array([0, 0, 0])
    p2 = np.array(volume.shape) - np.array(patch_size)
    assert np.all(p2 >= 0)
    z, x, y = np.random.uniform(p1, p2).round().astype(int)
    patch = volume[z:z + patch_size[0], x:x + patch_size[1], y:y + patch_size[2]]
    return patch


def center_crop(volume, patch_size):
    p2 = (np.array(volume.shape) - np.array(patch_size)) // 2
    patch = volume[p2[0]:p2[0] + patch_size[0], p2[1]:p2[1] + patch_size[1], p2[2]:p2[2] + patch_size[2]]
    return patch


def random_flip(patch):
    if random.random() > .5:
        patch = patch.flip(-1)
    return patch


def mask_decoder(mask, patient):
    temp = np.where(mask == 1)
    z1, x1, y1, z2, x2, y2 = temp[0].min(), temp[1].min(), temp[2].min(), temp[0].max(), temp[1].max(), temp[2].max()
    mask_size = [z2 - z1, y2 - y1, x2 - x1]
    mask_point = [z1, y1, x1]
    return np.array(mask_size), np.array(mask_point)


class AS_dataset(Dataset):
    def __init__(self, root, cross_validation, train, patch_size, k=None, seq='T2', online=True):

        root_dir = osp.join(root, 'train') if (train or k is not None) else osp.join(root, 'test')
        self.train = train

        if k:
            val_list = list(np.load(f'{cross_validation}/new_val{k}.npy'))
        print('exl:', 'label.xlsx')
        exl = pd.read_excel('label.xlsx',
                            sheet_name='Sheet1',
                            index_col=0,
                            header=0)
        self.patch_size = np.array(patch_size)
        self.online = online

        self.data = {}
        self.patients = []

        for patient in tqdm(sorted(os.listdir(root_dir))):
            neat_paitentID_dir = patient
            c = int(exl['label(axSpA:1;nonaxSpA:0)'][neat_paitentID_dir])
            e = int(exl['erosion'][neat_paitentID_dir])
            a = int(exl['ankylosis'][neat_paitentID_dir])
            if c == 1:
                label = 1
            elif c == 0:
                label = 0
            else:
                raise KeyError

            if a == 1:
                ankylosis = 1
            elif a == 0:
                ankylosis = 0
            else:
                raise KeyError

            if e == 1:
                erosion = 1
            elif e == 0:
                erosion = 0
            else:
                raise KeyError
            if k:
                if ((patient not in val_list) ^ train):
                    continue
            volume_path = osp.join(root_dir, patient, f'{seq}.npy')
            self.patients.append(patient)

            if online:
                self.data.update({patient: {
                    'volume_path': volume_path,
                    'label': label,
                    'erosion': erosion,
                    'ankylosis': ankylosis
                }})
            else:
                volume = np.load(volume_path).astype(np.float32)
                self.data.update({patient: {
                    'volume_path': volume,
                    'label': label,
                    'erosion': erosion,
                    'ankylosis': ankylosis
                }})
        print('Using MONAI transform')
        if train:
            self.transform = Compose([EnsureChannelFirst(channel_dim="no_channel"),
                                      RandFlip(spatial_axis=2, prob=.5),
                                      ScaleIntensityRangePercentiles(2, 99, -1, 1, clip=False),
                                      RandBiasField(degree=3, coeff_range=(0.0, 0.3), prob=0.2),
                                      RandGaussianSmooth(sigma_x=(0, 0), sigma_y=(.5, 5), sigma_z=(.5, 5), prob=.5),
                                      RandHistogramShift(num_control_points=10, prob=0.3),
                                      RandGibbsNoise(prob=0.2, alpha=(0.0, 1.0)),
                                      RandGaussianNoise(prob=.2, std=.5),
                                      RandAdjustContrast(prob=0.2, gamma=(0.1, 4.5)),
                                      RandSpatialCrop([12, 256, 512], random_size=False),
                                      ToTensor(dtype=torch.float)
                                      ])
        else:
            self.transform = Compose([EnsureChannelFirst(channel_dim="no_channel"),
                                      ScaleIntensityRangePercentiles(2, 99, -1, 1, clip=False),
                                      CenterSpatialCrop([12, 256, 512]),
                                      ToTensor(dtype=torch.float)
                                      ])

    def __getitem__(self, item):
        patient = self.patients[item]
        data = self.data[patient]
        label = data['label']
        erosion = data['erosion']
        ankylosis = data['ankylosis']
        if self.online:
            volume = np.load(data['volume_path']).astype(np.float32)
        else:
            volume = data['volume_path']
        assert isinstance(volume, np.ndarray)
        # data augmentation
        # patch = self.transform(volume)
        volume = normalization_mr(volume)
        if self.train:
            patch = random_crop(volume, self.patch_size)
            patch = torch.from_numpy(patch).to(torch.float32)
            patch = random_flip(patch)
        else:
            patch = center_crop(volume, self.patch_size)
            patch = torch.from_numpy(patch).to(torch.float32)
        assert patch.shape == tuple(self.patch_size)
        patch = patch.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        erosion = torch.tensor(erosion, dtype=torch.float).unsqueeze(0)
        ankylosis = torch.tensor(ankylosis, dtype=torch.float).unsqueeze(0)

        return {'patch': patch,
                'label': label,
                'erosion': erosion,
                'ankylosis': ankylosis
                }

    def __len__(self):
        return len(self.patients)


class AS_dataset_for_test(Dataset):
    def __init__(self, root, patch_size):
        root_dir = osp.join(root, 'test')

        self.patch_size = np.array(patch_size)
        self.data = {}
        self.patients = []
        self.seqs = ['FS', 'T1', 'T2']

        # images & labels

        exl = pd.read_excel('label.xlsx',
                            sheet_name='Sheet1',
                            index_col=0,
                            header=0)
        for patient in tqdm(os.listdir(root_dir)):

            neat_paitentID_dir = patient
            c = int(exl['label(axSpA:1;nonaxSpA:0)'][neat_paitentID_dir])
            if c == 1:
                label = 1
            elif c == 0:
                label = 0
            else:
                raise KeyError
            volume_path_dict = {}
            mask_path_dict = {}
            for seq in self.seqs:
                volume_path_dict[seq] = osp.join(root_dir, patient, f'{seq}.npy')
            self.patients.append(patient)
            self.data.update({patient: {
                'volume_path': volume_path_dict,
                'mask_path': mask_path_dict,
                'label': label
            }})

    def __getitem__(self, item):
        patient = self.patients[item]
        data = self.data[patient]
        volumes = dict()
        for seq in self.seqs:
            volumes[seq] = np.load(data['volume_path'][seq]).astype(np.float32)
        label = data['label']
        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)

        patch_dict = {}
        for seq in self.seqs:
            volume = volumes[seq]
            volume = normalization_mr(volume)
            patch = center_crop(volume, self.patch_size)
            assert patch.shape == tuple(self.patch_size)
            patch = torch.from_numpy(patch).to(torch.float32)
            # patch = random_flip(patch)
            patch = patch.unsqueeze(0)
            patch_dict.update({seq: patch})

        return {'patch': patch_dict,
                'label': label,
                'patient': patient}

    def __len__(self):
        return len(self.patients)


class AS_dataset_for_test_and_train(Dataset):
    def __init__(self, root, patch_size):

        self.patch_size = np.array(patch_size)
        self.data = {}
        self.patients = []
        self.seqs = ['FS', 'T1', 'T2']

        # images & labels

        exl = pd.read_excel('label.xlsx',
                            sheet_name='Sheet1',
                            index_col=0,
                            header=0)
        for mode in ['test']:  # 'train', 'test'
            root_dir = osp.join(root, mode)
            for patient in tqdm(os.listdir(root_dir)):
                neat_paitentID_dir = patient
                c = int(exl['label(axSpA:1;nonaxSpA:0)'][neat_paitentID_dir])
                e = int(exl['erosion'][neat_paitentID_dir])
                a = int(exl['ankylosis'][neat_paitentID_dir])
                if c == 1:
                    label = 1
                elif c == 0:
                    label = 0
                else:
                    raise KeyError

                if a == 1:
                    ankylosis = 1
                elif a == 0:
                    ankylosis = 0
                else:
                    raise KeyError

                if e == 1:
                    erosion = 1
                elif e == 0:
                    erosion = 0
                else:
                    raise KeyError
                volume_path_dict = {}
                mask_path_dict = {}
                for seq in self.seqs:
                    volume_path_dict[seq] = osp.join(root_dir, patient, f'{seq}.npy')
                self.patients.append(patient)
                self.data.update({patient: {
                    'volume_path': volume_path_dict,
                    'mask_path': mask_path_dict,
                    'label': label,
                    'erosion': erosion,
                    'ankylosis': ankylosis
                }})

    def __getitem__(self, item):
        patient = self.patients[item]
        data = self.data[patient]
        volumes = dict()
        for seq in self.seqs:
            volumes[seq] = np.load(data['volume_path'][seq]).astype(np.float32)
        label = data['label']
        erosion = data['erosion']
        ankylosis = data['ankylosis']
        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        erosion = torch.tensor(erosion, dtype=torch.float).unsqueeze(0)
        ankylosis = torch.tensor(ankylosis, dtype=torch.float).unsqueeze(0)

        patch_dict = {}
        for seq in self.seqs:
            volume = volumes[seq]
            volume = normalization_mr(volume)
            patch = center_crop(volume, self.patch_size)
            assert patch.shape == tuple(self.patch_size)
            patch = torch.from_numpy(patch).to(torch.float32)
            # patch = random_flip(patch)
            patch = patch.unsqueeze(0)
            patch_dict.update({seq: patch})

        return {'patch': patch_dict,
                'label': label,
                'patient': patient,
                'erosion': erosion,
                'ankylosis': ankylosis
                }

    def __len__(self):
        return len(self.patients)
