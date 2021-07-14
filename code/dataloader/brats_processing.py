import os
import numpy as np
import SimpleITK as sitk
import glob
from scipy.ndimage import interpolation
import h5py


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

def mask2bbox(image, mask, pert=[0, 0, 0]):
    boundary = np.array(np.where(mask > 0)).transpose(1, 0)
    bbox_index = []
    for i in range(len(mask.shape)):
        index = (max(0, boundary[:, i].min() - pert[i]),
                 min(mask.shape[i], boundary[:, i].max() + pert[i]))
        bbox_index.append(index)
    print(image.shape, mask.shape)
    print(bbox_index)
    cropped_image = image[bbox_index[0][0]:bbox_index[0][1], bbox_index[1][0]:bbox_index[1][1],
                          bbox_index[2][0]:bbox_index[2][1]]
    cropped_mask = mask[bbox_index[0][0]:bbox_index[0][1], bbox_index[1][0]:bbox_index[1][1],
                        bbox_index[2][0]:bbox_index[2][1]]

    if cropped_image.shape == cropped_mask.shape:
        return cropped_image, cropped_mask
    else:
        print("Error")


mask_path = glob.glob(
    "../data/brats18/*/*_flair.nii.gz")
for case in mask_path:
    msk_itk = sitk.ReadImage(case)
    origin = msk_itk.GetOrigin()
    spacing = msk_itk.GetSpacing()
    direction = msk_itk.GetDirection()
    mask = sitk.GetArrayFromImage(msk_itk)

    img_path = case.replace("flair", "seg")
    if os.path.exists(img_path):
        img_itk = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(img_itk)
        cropped_mask, cropped_image = mask2bbox(image, mask)
        cropped_image = MedicalImageDeal(cropped_image, percent=0.999).valid_img
        cropped_mask[cropped_mask > 0] = 1
        print(cropped_image.shape)
        cropped_image = cropped_image.astype(np.float32)
        item = case.split("/")[-1].split(".")[0].replace("_flair", "")
        name = "../data/processed_brats18/" + item
        cropped_img_itk = sitk.GetImageFromArray(cropped_image)
        cropped_lab_itk = sitk.GetImageFromArray(cropped_mask)
        cropped_img_itk.SetOrigin(origin)
        cropped_img_itk.SetSpacing(spacing)
        cropped_img_itk.SetDirection(direction)
        cropped_lab_itk.SetOrigin(origin)
        cropped_lab_itk.SetSpacing(spacing)
        cropped_lab_itk.SetDirection(direction)
        sitk.WriteImage(cropped_img_itk, name + "_img.nii.gz")
        sitk.WriteImage(cropped_lab_itk, name + "_lab.nii.gz")
