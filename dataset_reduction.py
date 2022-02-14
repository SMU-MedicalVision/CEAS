import os
import os.path as osp
from tqdm import tqdm
from glob import glob
import numpy as np
import skimage.measure as measure

root_dir = '../AS_Dataset/npy_exp1'
target_dir = '../AS_Dataset/npy_exp1_thin'

seqs = ['FS', 'T1', 'T2']


def mask_decoder1(mask, patient):
    # label, num = measure.label(mask, return_num=True)
    # region = measure.regionprops(label)
    # if num != 1:
    #     region = sorted(region, key=lambda x: x.area, reverse=True)
    #     print(f'Multi-regions detected in {patient}, the second largest area {region[1].area}!')
    # z1, y1, x1, z2, y2, x2 = region[0].bbox
    temp = np.where(mask == 1)
    z1, x1, y1, z2, x2, y2 = temp[0].min(), temp[1].min(), temp[2].min(), temp[0].max(), temp[1].max(), temp[2].max()
    mask_size = [z2 - z1 + 1, x2 - x1 + 1, y2 - y1 + 1]
    mask_point = [z1, x1, y1]
    return np.array(mask_size), np.array(mask_point)


def mask_decoder2(mask, patient):
    label, num = measure.label(mask, return_num=True)
    region = measure.regionprops(label)
    if num != 1:
        region = sorted(region, key=lambda x: x.area, reverse=True)
        print(f'Multi-regions detected in {patient}, the second largest area {region[1].area}!')
    z1, y1, x1, z2, y2, x2 = region[0].bbox
    mask_size = [z2 - z1, y2 - y1, x2 - x1]
    mask_point = [z1, y1, x1]
    return np.array(mask_size), np.array(mask_point)


def random_crop_with_mask(volume, mask_point, mask_size, patch_size, patient):
    patch_size = np.array(patch_size)
    image_size = np.array(volume.shape)

    flag = patch_size > mask_size
    patch_shape = np.max([mask_size, patch_size], axis=0) + flag * 2 * np.abs(patch_size - mask_size)
    patch_origin = mask_point - flag * np.abs(patch_size - mask_size)

    p1 = np.max([[0., 0., 0.], patch_origin], axis=0).astype(np.int)
    p2 = np.min([image_size, p1 + patch_shape], axis=0).astype(np.int)
    patch = volume[p1[0]:p2[0], p1[1]:p2[1], p1[2]:p2[2]]
    assert patch.shape[0] >= 12, (patient, patch.shape)
    assert patch.shape[1] >= 256, (patient, patch.shape)
    assert patch.shape[2] >= 512, (patient, patch.shape)
    return patch


patch_size = [12, 256, 512]

for patient in glob(osp.join(root_dir, '*/*/*')):
    target_patient_dir = patient.replace('npy_exp1', 'npy_exp1_thin')
    os.makedirs(target_patient_dir)
    print(patient)
    for seq in seqs:
        # load npy
        volume_path = osp.join(patient, f'{seq}.npy')
        mask_path = osp.join(patient, f'{seq}_mask.npy')
        volume = np.load(volume_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # operation
        if volume.shape[0] < 12:
            new_volume = np.zeros([12, volume.shape[1], volume.shape[2]], dtype=volume.dtype)
            new_volume[:volume.shape[0]] = volume
            new_mask = np.zeros([12, volume.shape[1], volume.shape[2]], dtype=mask.dtype)
            new_mask[:volume.shape[0]] = mask
            volume = new_volume
            mask = new_mask
        mask_size1, mask_point1 = mask_decoder1(mask, patient)
        mask_size2, mask_point2 = mask_decoder2(mask, patient)
        assert not np.any(mask_size2 - mask_size1), patient
        assert not np.any(mask_point2 - mask_point1), patient
        patch = random_crop_with_mask(volume, mask_point1, mask_size1, patch_size, patient)
        print(mask_size2, patch.shape)

        # save new .npy
        volume_path = volume_path.replace('npy_exp1', 'npy_exp1_thin')
        np.save(volume_path, patch)
