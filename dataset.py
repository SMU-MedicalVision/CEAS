import os
import SimpleITK as sitk
import os.path as osp
import numpy as np
import Nii_utils
import torch
from torch.utils.data import Dataset
import skimage.measure as measure
import random
from tqdm import tqdm
import pandas as pd
import warnings
import miceforest as mf
import time

warnings.filterwarnings('ignore')

def normalization_mr(X):
    p2, p99 = np.percentile(X, (2, 99))
    result = ((X - p2) / (p99 - p2)).astype('float')
    return result * 2 - 1

def normalization_mr_vit(X):
    p2, p99 = np.percentile(X, (2, 99))
    result = ((X - p2) / (p99 - p2)).astype('float')
    return result


def random_crop_with_mask(volume, mask_point, mask_size, patch_size, patient):
    flag = mask_size > patch_size
    new_mask_size = patch_size * flag + mask_size * ~flag
    differential = new_mask_size - mask_size

    mask_point -= (differential / 2).round().astype(np.int)
    image_size = np.array(volume.shape)

    p1 = np.max([[0., 0., 0.], mask_point + new_mask_size - patch_size], axis=0)
    p2 = np.min([mask_point, image_size - patch_size], axis=0)
    z, x, y = np.random.uniform(p1, p2).round().astype(np.int)
    patch = volume[z:z + patch_size[0], x:x + patch_size[1], y:y + patch_size[2]]
    assert patch.shape[0] == 12, (patient, patch.shape)
    assert patch.shape[1] == 256, (patient, patch.shape)
    assert patch.shape[2] == 512, (patient, patch.shape)
    return patch

def random_crop(volume, patch_size):
    p1 = np.array([0, 0, 0])
    p2 = np.array(volume.shape) - np.array(patch_size)
    assert np.all(p2 >= 0)
    z, x, y = np.random.uniform(p1, p2).round().astype(np.int)
    patch = volume[z:z + patch_size[0], x:x + patch_size[1], y:y + patch_size[2]]
    return patch

def center_crop(volume, patch_size):
    p2 = (np.array(volume.shape) - np.array(patch_size)) // 2
    patch = volume[p2[0]:p2[0] + patch_size[0], p2[1]:p2[1] + patch_size[1], p2[2]:p2[2] + patch_size[2]]
    return patch

def random_flip(patch):
    if random.random() > .5:
        patch = patch.flip(2)
    return patch

def mask_decoder(mask, patient):
    temp = np.where(mask == 1)
    z1, x1, y1, z2, x2, y2 = temp[0].min(), temp[1].min(), temp[2].min(), temp[0].max(), temp[1].max(), temp[2].max()
    mask_size = [z2 - z1, y2 - y1, x2 - x1]
    mask_point = [z1, y1, x1]
    return np.array(mask_size), np.array(mask_point)


class AS_dataset(Dataset):
    def __init__(self, root, cross_validation, train, patch_size, k=None, use_CF=False, seq='T2', online=True):

        root_dir = osp.join(root, 'train') if (train or k is not None) else osp.join(root, 'test')
        if k is not None:
            val_list = list(np.load(f'{cross_validation}/new_val{k}.npy'))

        self.patch_size = np.array(patch_size)
        self.use_CF = use_CF
        self.online = online

        self.data = {}
        self.patients = []

        # images & labels
        for c in os.listdir(root_dir):
            if c == 'nonAS':
                label = 0
            elif c == 'AS':
                label = 1
            else:
                raise KeyError
            class_dir = osp.join(root_dir, c)
            for patient in tqdm(os.listdir(class_dir)):
                # if not test:
                if k is not None:
                    if ((patient not in val_list) ^ train):
                        continue
                volume_path = osp.join(class_dir, patient, f'{seq}.npy')
                self.patients.append(patient)

                if online:
                    self.data.update({patient: {
                        'volume_path': volume_path,
                        'label': label
                    }})
                else:
                    volume = np.load(volume_path).astype(np.float32)
                    self.data.update({patient: {
                        'volume_path': volume,
                        'label': label
                    }})

    def iteration(self, epoch):
        self.df_concat = pd.read_excel(f'./log_excels/{epoch}.xlsx', sheet_name='Sheet1', index_col=0,
                                       header=0)

    def __getitem__(self, item):
        patient = self.patients[item]
        data = self.data[patient]
        label = data['label']
        if self.online:
            volume = np.load(data['volume_path']).astype(np.float32)
        else:
            volume = data['volume_path']
        volume = normalization_mr(volume)
        patch = random_crop(volume, self.patch_size)
        assert patch.shape == tuple(self.patch_size)
        patch = torch.from_numpy(patch).to(torch.float32)
        patch = random_flip(patch)
        patch = patch.unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)

        # clinical information
        if self.use_CF:
            series = self.df_concat.loc[patient.split('-')[0]]
            gender = series['Gender']
            age = series['Age'] / 100.
            hla = series['HLA-B27']

            CF = torch.tensor([hla,gender,age], dtype=torch.float)
            return {'CF': CF,
                    'patch': patch,
                    'label': label}
        else:
            return {'patch': patch,
                    'label': label}

    def __len__(self):
        return len(self.patients)


class AS_dataset_for_test(Dataset):
    def __init__(self, root, patch_size, use_CF=False):
        root_dir = osp.join(root, 'test')

        self.patch_size = np.array(patch_size)
        self.use_CF = use_CF

        self.data = {}
        self.patients = []
        self.seqs = ['FS', 'T1', 'T2']

        # images & labels
        for c in os.listdir(root_dir):
            if c == 'nonAS':
                label = 0
            elif c == 'AS':
                label = 1
            else:
                raise KeyError
            class_dir = osp.join(root_dir, c)
            for patient in tqdm(os.listdir(class_dir)):
                volume_path_dict = {}
                mask_path_dict = {}
                for seq in self.seqs:
                    volume_path_dict[seq] = osp.join(class_dir, patient, f'{seq}.npy')
                    mask_path_dict[seq] = osp.join(class_dir, patient, f'{seq}_mask.npy')

                self.patients.append(patient)
                self.data.update({patient: {
                    'volume_path': volume_path_dict,
                    'mask_path': mask_path_dict,
                    'label': label
                }})

        # clinical information
        if use_CF:
            exl = pd.read_excel('AS-非AS-健康人诊断与鉴别影像与临床信息汇总20210723.xlsx', sheet_name=None, index_col=2, header=1)
            df_concat = pd.concat([exl['AS'], exl['non-AS'], exl['health']], join='inner')
            df_concat.index = df_concat.index.map(lambda i: i.split('-')[0])
            self.CFs = {}

    def iteration(self, epoch):
        self.df_concat = pd.read_excel(f'./log_excels/{epoch}.xlsx', sheet_name='Sheet1', index_col=0,
                                       header=0)

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

        # dict construction ??**
        if self.use_CF:
            series = self.df_concat.loc[patient.split('-')[0]]
            gender = series['Gender']
            age = series['Age'] / 100.
            hla = series['HLA-B27']

            CF = torch.tensor([hla,gender,age], dtype=torch.float)
            return {'CF': CF,
                    'patch': patch_dict,
                    'label': label,
                    'patient': patient}
        else:
            return {'patch': patch_dict,
                    'label': label,
                    'patient': patient}

    def __len__(self):
        return len(self.patients)

class AS_dataset_for_test_val(Dataset):
    def __init__(self, root, patch_size,cross_validation,k, use_CF=False):
        root_dir = osp.join(root, 'train')
        val_list = list(np.load(f'{cross_validation}/new_val{k}.npy'))

        self.patch_size = np.array(patch_size)
        self.use_CF = use_CF

        self.data = {}
        self.patients = []
        self.seqs = ['FS', 'T1', 'T2']

        # images & labels
        for c in os.listdir(root_dir):
            if c == 'nonAS':
                label = 0
            elif c == 'AS':
                label = 1
            else:
                raise KeyError
            class_dir = osp.join(root_dir, c)
            for patient in tqdm(os.listdir(class_dir)):
                if patient not in val_list:
                    continue
                volume_path_dict = {}
                mask_path_dict = {}
                for seq in self.seqs:
                    volume_path_dict[seq] = osp.join(class_dir, patient, f'{seq}.npy')
                    mask_path_dict[seq] = osp.join(class_dir, patient, f'{seq}_mask.npy')

                self.patients.append(patient)
                self.data.update({patient: {
                    'volume_path': volume_path_dict,
                    'mask_path': mask_path_dict,
                    'label': label
                }})

        # clinical information
        if use_CF:
            exl = pd.read_excel('AS-非AS-健康人诊断与鉴别影像与临床信息汇总20210723.xlsx', sheet_name=None, index_col=2, header=1)
            df_concat = pd.concat([exl['AS'], exl['non-AS'], exl['health']], join='inner')
            df_concat.index = df_concat.index.map(lambda i: i.split('-')[0])
            self.CF = {}

    def iteration(self, epoch):
        self.df_concat = pd.read_excel(f'./log_excels/{epoch}.xlsx', sheet_name='Sheet1', index_col=0,
                                       header=0)

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

        # dict construction ??**
        if self.use_CF:
            series = self.df_concat.loc[patient.split('-')[0]]
            gender = series['Gender']
            age = series['Age'] / 100.
            hla = series['HLA-B27']

            CF = torch.tensor([hla,gender,age], dtype=torch.float)
            return {'CF': CF,
                    'patch': patch_dict,
                    'label': label,
                    'patient': patient}
        else:
            return {'patch': patch_dict,
                    'label': label,
                    'patient': patient}

    def __len__(self):
        return len(self.patients)



class AS_dataset_logistic(Dataset):
    def __init__(self, root, train, cross_validation=None, k=None):
        root_dir = osp.join(root, 'train') if (train or k is not None) else osp.join(root, 'test')
        if k is not None:
            val_list = list(np.load(f'{cross_validation}/new_val{k}.npy'))

        self.data = {}
        self.patients = []

        # images & labels
        for c in os.listdir(root_dir):
            if c == 'nonAS':
                label = 0
            elif c == 'AS':
                label = 1
            else:
                raise KeyError
            class_dir = osp.join(root_dir, c)
            for patient in tqdm(os.listdir(class_dir)):
                # if not test:
                if k is not None:
                    if ((patient not in val_list) ^ train):
                        continue

                self.patients.append(patient)
                self.data.update({patient: {
                    'label': label
                }})

    def iteration(self, epoch):
        self.df_concat = pd.read_excel(f'./log_excels/{epoch}.xlsx', sheet_name='Sheet1', index_col=0,
                                       header=0)

    def __getitem__(self, item):
        patient = self.patients[item]
        data = self.data[patient]
        label = data['label']

        series = self.df_concat.loc[patient.split('-')[0]]
        gender = series['Gender']
        age = series['Age'] / 100.
        duration = series['Disease Duration'] / 100.
        esr = series['ESR'] / 100.
        crp = series['CRP'] / 100.
        hla = series['HLA-B27']

        CF = torch.tensor([gender, age, duration, esr, crp, hla], dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        return {'patch': CF,
                'label': label}

    def __len__(self):
        return len(self.patients)
