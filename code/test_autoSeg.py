import os
import argparse
import torch
import h5py
import numpy as np
from medpy import metric
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import nibabel as nib
from networks.unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2018', help='Name of Experiment')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
test_save_path = "../data/BraTS2018/autoSeg/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2
patch_size = [96, 96, 96]
with open(FLAGS.root_path + '/test.txt', 'r') as f:
    image_list = f.readlines()
image_list = sorted([item.replace('\n', '').split(".")[0]
                     for item in image_list])


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
    std = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def nifty2array(path):
    img_itk = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img_itk)
    return data


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    ravd = metric.binary.ravd(pred, gt)
    assd = metric.binary.assd(pred, gt)
    return dice, hd95, abs(ravd), assd


def test_all_case(net, image_list, patch_size, save_result, test_save_path):
    total_metric = 0.0
    for id in tqdm(image_list):
        print(id)
        img_path = FLAGS.root_path + "/data/{}_img.nii.gz".format(id)
        lab_path = FLAGS.root_path + "/data/{}_lab.nii.gz".format(id)
        raw_image = nifty2array(img_path)
        label = nifty2array(lab_path)
        spacingzxy = [patch_size[i] / raw_image.shape[i] for i in range(3)]
        norm_raw_image = itensity_normalize_one_volume(raw_image)
        image = zoom(norm_raw_image, spacingzxy, order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        output = net(inputs)
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output, dim=1).squeeze(0)
        prediction = output.cpu().data.numpy()
        new_spacingzxy = [raw_image.shape[i] /
                          prediction.shape[i] for i in range(3)]
        prediction = zoom(prediction, new_spacingzxy, order=0)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        print(single_metric)
        total_metric += np.asarray(single_metric)

        if save_result:
            sitk.WriteImage(sitk.GetImageFromArray(prediction.astype(
                np.float32)), test_save_path + id + "_seg.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(raw_image[:].astype(
                np.float32)), test_save_path + id + "_img.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(label[:].astype(
                np.float32)), test_save_path + id + "_lab.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_calculate_metric(save_mode_path):
    net = UNet(in_chns=1, class_num=2).cuda()
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, patch_size=patch_size,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    pth = "../model/autoSeg/iter_30000.pth"
    metric = test_calculate_metric(pth)
