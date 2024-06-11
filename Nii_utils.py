import SimpleITK as sitk
import numpy as np
import pydicom
import re
import os
import os.path as osp
import time

def NiiDataRead(path):
    start=time.time()
    nii = sitk.ReadImage(path)
    # print(time.time()-start)
    spacing = nii.GetSpacing()  # [x,y,z]
    volumn = sitk.GetArrayFromImage(nii)  # [z,y,x]

    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    spacing_ = np.array([spacing_z, spacing_y, spacing_x])
    return volumn.astype(np.float32), spacing_.astype(np.float32)


def NiiDataWrite(save_path, volumn, spacing):
    spacing = spacing.astype(np.float64)
    raw = sitk.GetImageFromArray(volumn)
    spacing_ = (spacing[2], spacing[1], spacing[0])
    raw.SetSpacing(spacing_)
    sitk.WriteImage(raw, save_path)


def N4BiasFieldCorrection(volumn_path, save_path):  # ,mask_path,save_path):
    img = sitk.ReadImage(volumn_path)
    # mask,_ = sitk.ReadImage(mask_path)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    inputVolumn = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    sitk.WriteImage(corrector.Execute(inputVolumn, mask), save_path)


def DCM2NII(DCM_DIR, OUT_PATH):
    """
    :param DCM_DIR: Input folder set to be converted.
    :param OUT_PATH: Output file suffixed with .nii.gz . *(Relative path)
    :return: No retuen.
    """
    fuse_list = []
    for dicom_file in os.listdir(DCM_DIR):
        dicom = pydicom.dcmread(osp.join(DCM_DIR, dicom_file))
        fuse_list.append([dicom.pixel_array, float(dicom.SliceLocation)])
    # 按照每层位置(Z轴方向)由小到大排序
    fuse_list.sort(key=lambda x: x[1])
    volume_list = [i[0] for i in fuse_list]
    volume = np.array(volume_list).astype(np.float32) - 1024
    [spacing_x, spacing_y] = dicom.PixelSpacing
    spacing = np.array([dicom.SliceThickness, spacing_x, spacing_y])
    NiiDataWrite(OUT_PATH, volume, spacing)

# 4dicom转nii调参数
def DCM2NII2(path_read, path_save):
    '''
    :param path_read: path to .dcms or a list of .dcm paths
    :param path_save: path to converted nii file
    '''
    if isinstance(path_read, str):
        # GetGDCMSeriesIDs读取序列号相同的dcm文件
        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
        # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    elif isinstance(path_read, list):
        series_file_names = path_read
        # 按照每层位置(Z轴方向)由小到大排序
        fuse_list = []
        for dicom_file in series_file_names:
            dicom = pydicom.dcmread(dicom_file)
            fuse_list.append([dicom_file, float(dicom.InstanceNumber)])
        fuse_list.sort(key=lambda x: x[1])
        series_file_names = [i[0] for i in fuse_list]
    else:
        raise TypeError


    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save)

    spacing = image3d.GetSpacing()  # [x,y,z]
    volumn = sitk.GetArrayFromImage(image3d)  # [z,y,x]

    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    spacing_ = np.array([spacing_z, spacing_y, spacing_x])

    return len(series_file_names), volumn.shape, spacing_.astype(np.float32)

def DCM2NII_MRI(DCM_DIR, OUT_PATH):
    """

    :param DCM_DIR: Input folder set to be converted.
    :param OUT_PATH: Output file suffixed with .nii.gz . *(Relative path)
    :return: No retuen.
    """
    fuse_list = []
    for dicom_file in os.listdir(DCM_DIR):
        dicom = pydicom.dcmread(osp.join(DCM_DIR, dicom_file))
        file = sitk.ReadImage(osp.join(DCM_DIR, dicom_file))
        data = sitk.GetArrayFromImage(file)
        fuse_list.append([data[0], float(dicom.ImagePositionPatient[2])])
    # 按照每层位置(Z轴方向)由小到大排序
    fuse_list.sort(key=lambda x: x[1])
    volume_list = [i[0] for i in fuse_list]
    volume = np.array(volume_list).astype(np.float32)
    [spacing_x, spacing_y] = dicom.PixelSpacing
    spacing_z = dicom.SpacingBetweenSlices if hasattr(dicom, 'SpacingBetweenSlices') else dicom.SliceThickness
    spacing = np.array([spacing_z, spacing_x, spacing_y])
    NiiDataWrite(OUT_PATH, volume, spacing)


def modality_encoder(mr_dir):
    """

    :param mr_dir: Directory path to the MRI DICOM directory which you wanna get the modality of.
    :return: modality
    """
    # mr_file = os.listdir(mr_dir)[0]
    mr_file = osp.join(mr_dir, os.listdir(mr_dir)[0])
    dcm = pydicom.read_file(mr_file)
    sname = dcm.SeriesDescription
    pname = dcm.ProtocolName if hasattr(dcm, 'ProtocolName') else None
    if (pname is None) or ('/' in pname):
        pname = sname
    name = sname + ' ' + pname
    name = name.upper()
    con = dcm.ContrastBolusAgent if hasattr(dcm, 'ContrastBolusAgent') else None

    if 'T1' in name:
        modality = 'T1'
    elif 'T2' in name:
        modality = 'T2'
    elif ('LOC' in name) or sname == '':
        modality = 'LOC'
        # print([sname, pname, con], modality)
        return modality
    else:
        modality = ''

    if ('AX' in name) or ('TRA' in name):
        modality = 'AX ' + modality
    elif ('SG' in name) or ('SAG' in name):
        modality = 'SG ' + modality
    elif ('CO' in name) or ('COR' in name):
        modality = 'CO ' + modality
    else:
        modality = 'AX ' + modality
        # print([sname, pname, con])

    if ('FSE' in name) or ('TSE' in name):
        modality = modality + ' FSE'

    if 'FS' in re.split('[ _+-]', name):
        modality = modality + ' FS'

    if 'IR' in re.split('[ _+-]', name):
        modality = modality + ' IR'

    if 'FLAIR' in name:
        modality = modality + ' FLAIR'

    if 'STIR' in name:
        modality = modality + ' STIR'

    if 'DARK' in name:
        modality += ' DARK-FLUID'

    if (con is not None) or ('C' in re.split('[ _+-]', name)):
        modality = modality + ' +C'
    # print([sname, pname, con],modality)
    return modality


def dcmDataRead(path):
    dicom = pydicom.dcmread(path)
    file = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(file)
    [spacing_x, spacing_y] = dicom.PixelSpacing
    spacing_z = dicom.SpacingBetweenSlices if hasattr(dicom, 'SpacingBetweenSlices') else dicom.SliceThickness
    spacing_ = np.array([spacing_z, spacing_x, spacing_y])
    volumn = data[0]

    return volumn.astype(np.float32), spacing_.astype(np.float32)


def raw2nii(mhd_path, save_path):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    data = sitk.ReadImage(mhd_path)
    sitk.WriteImage(data, save_path)
