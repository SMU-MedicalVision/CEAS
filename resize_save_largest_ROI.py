import os
import os.path as osp
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage.measure as measure
source_dir = '../AS_Dataset/nii_3seq'
target_dir = '../AS_Dataset/resized_3seq'

def resize(volume, mask, new_xy_spacing):
    # spacing & size calculation
    original_spacing = np.array(volume.GetSpacing(), float)
    if abs(original_spacing[1] - new_xy_spacing) < 0.05:
        return volume, mask
    new_spacing = np.array([new_xy_spacing, new_xy_spacing, original_spacing[2]], float)
    original_size = volume.GetSize()
    factor = new_spacing / original_spacing
    new_size = (original_size / factor).astype(np.int)

    # resampler setting
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))

    # Execute
    resampler.SetInterpolator(sitk.sitkLinear)
    resized_volume = resampler.Execute(volume)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resized_mask = resampler.Execute(mask)
    return resized_volume, resized_mask


def save_largest_mask(mask):
    label, num = measure.label(mask, return_num=True)
    region = measure.regionprops(label)
    if num != 1:
        region = sorted(region, key=lambda x: x.area, reverse=True)
        print(f'Multi-regions detected in {patient}, the second largest area {region[1].area}!')
    largest_ROI = (label == region[0].label).astype(np.uint16)
    return largest_ROI




ignored = 0
seqs=['FS','T1','T2']
for c in os.listdir(source_dir):
    print(c)
    class_dir = osp.join(source_dir, c)
    output_dir = osp.join(target_dir, c)
    os.makedirs(output_dir, exist_ok=True)
    for patient in os.listdir(class_dir):
        if patient != 'MR24238':
            continue
        for seq in seqs:
            if seq != 'T2':
                continue
            patient_dir = osp.join(class_dir, patient)
            volume = sitk.ReadImage(osp.join(patient_dir, f'{seq}.nii.gz'))
            mask = sitk.ReadImage(osp.join(patient_dir, f'{seq}_mask.nii.gz'))
            ROI = sitk.GetArrayFromImage(mask)
            largest_ROI = save_largest_mask(ROI)
            largest_ROI = sitk.GetImageFromArray(largest_ROI)
            largest_ROI.CopyInformation(mask)
            volume, mask = resize(volume, largest_ROI, 0.4)
            if seq == 'FS':
                os.makedirs(osp.join(output_dir, patient))
            sitk.WriteImage(volume, osp.join(output_dir, patient, f'{seq}.nii.gz'))
            sitk.WriteImage(mask, osp.join(output_dir, patient, f'{seq}_mask.nii.gz'))
print(f'Ignored {ignored} patients')
