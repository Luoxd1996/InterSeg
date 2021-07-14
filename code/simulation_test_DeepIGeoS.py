import argparse
import logging
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
from medpy import metric
from scipy.ndimage import interpolation

from networks.unet import UNet
from simulate_interactions import generate_simulation_deepigeos_validation

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2018', help='Name of Experiment')
parser.add_argument('--max_iteration', type=int, default=15,
                    help='maximum iteration number to validation')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
args = parser.parse_args()

patch_size = [96, 96, 96]
num_classes = 2


max_iteration = args.max_iteration


def nifity_to_array(path):
    itk_data = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(itk_data)
    return array


def cal_metric(gt, pred):
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        assd = metric.binary.assd(pred, gt)
        return np.array([dice, assd])
    else:
        return np.zeros(2)


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


def ReScale(output_size, image, order):
    (w, h, d) = image.shape
    spacingzxy = [output_size[0] / w,
                  output_size[1] / h, output_size[2] / d]
    zoomed_image = interpolation.zoom(image, spacingzxy, order=order)
    return zoomed_image


def interSeg_test(autoCNN, interCNN, data_list, max_val_iteration, save_results=None):
    autoCNN.eval()
    interCNN.eval()
    autoCNN_results = np.zeros((len(data_list), 2))
    interCNN_results = np.zeros((len(data_list), max_val_iteration, 2))
    num = 0
    for case in data_list:
        image = nifity_to_array(case[0])
        label = nifity_to_array(case[1])
        image = itensity_normalize_one_volume(image)
        zoomed_image = ReScale([96, 96, 96], image, order=3)
        zoomed_label = ReScale([96, 96, 96], label, order=0)
        sitk.WriteImage(sitk.GetImageFromArray(
            zoomed_label), "zoomed_label.nii.gz")
        autoInput = torch.from_numpy(
            zoomed_image).unsqueeze(0).unsqueeze(0).cuda()
        outputs = autoCNN(autoInput)
        autoOutput_soft = torch.softmax(outputs, dim=1)
        autoSeg = torch.argmax(autoOutput_soft, dim=1).squeeze(0)
        autoSeg = autoSeg.cpu().data.numpy().astype(np.float32)
        new_spacingzxy = [image.shape[i] /
                          autoSeg.shape[i] for i in range(3)]
        autoResult = interpolation.zoom(autoSeg, new_spacingzxy, order=0)
        autoResult = cal_metric(label, autoResult)
        print("autoResult", autoResult)
        autoCNN_results[num, :] = autoResult
        seed_org = torch.zeros_like(autoOutput_soft)
        zoomed_label = torch.from_numpy(zoomed_label).unsqueeze(0).cuda()
        for item in range(max_val_iteration):
            [inputs, geos_maps, seeds] = generate_simulation_deepigeos_validation(autoInput, zoomed_label, outputs,
                                                                                  seed_org, transform_type="geodesic", distance_threshold=0.4)
            outputs = interCNN(inputs.cuda())
            outputs_soft = torch.softmax(outputs, dim=1)
            interSeg = torch.argmax(outputs_soft, dim=1).squeeze(0)
            interSeg = interSeg.cpu().data.numpy()
            sitk.WriteImage(sitk.GetImageFromArray(interSeg.astype(
                np.float32)), "round_{}_pred.nii.gz".format(item))
            new_spacingzxy = [image.shape[i] /
                              interSeg.shape[i] for i in range(3)]
            interResult = interpolation.zoom(interSeg, new_spacingzxy, order=0)
            interResult = cal_metric(label, interResult)
            print("Round: {} results".format(item), interResult)
            interCNN_results[num, item, :] = interResult
            seed_org += seeds
        num += 1
    np.save("autoCNN_results.npy", autoCNN_results)
    np.save("interCNN_results.npy", interCNN_results)
    return "Finished"


if __name__ == '__main__':
    autopth = "../model/BraTS2018/autoSeg/iter_30000.pth"
    interpth = "../model/BraTS2018/DeepIGeoS/iter_30000.pth"
    autoCNN = UNet(in_chns=1, class_num=2).cuda()
    autoCNN.load_state_dict(torch.load(
        "../model/BraTS2018/autoSeg/iter_30000.pth"))
    interCNN = UNet(in_chns=4, class_num=2).cuda()
    interCNN.load_state_dict(torch.load(
        "../model/BraTS2018/DeepIGeoS/iter_30000.pth"))

    with open(args.root_path + '/test.txt', 'r') as f:
        sample_list = f.readlines()
    sample_list = [item.replace('\n', '').split(
        ".")[0] for item in sample_list]
    data_list = [[args.root_path + "/data/{}_img.nii.gz".format(
        image_name), args.root_path + "/data/{}_lab.nii.gz".format(image_name)] for image_name in sample_list]

    interSeg_test(autoCNN, interCNN, data_list, max_val_iteration=10)
