from glob import glob
import numpy as np
import os
import os.path as osp
import SimpleITK as sitk
from tqdm import tqdm
import shutil
for nii in tqdm(glob(r'../AS_Dataset/*/*/*/*.nii.gz')):

    npy_dir = osp.dirname(nii).replace('splited_exp1','npy_exp1')
    npy_base = osp.basename(nii).replace('nii.gz','npy')
    if not osp.exists(osp.join(npy_dir,npy_base)):
        os.makedirs(npy_dir,exist_ok=True)

        volume =sitk.GetArrayFromImage(sitk.ReadImage(nii)).astype(np.float32)
        if 'mask' in nii:
            volume = volume.astype(np.bool)
        np.save(osp.join(npy_dir,npy_base),volume)
