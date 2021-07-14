import os
import random
import torch
import numpy as np
import SimpleITK as sitk
import skimage
import skimage.measure as measure
from scipy import ndimage
import GeodisTK
from torch import threshold
from skimage.morphology import (remove_small_holes, remove_small_objects, closing,
                                opening, erosion, dilation, disk, square, diamond, star, cube, octahedron, ball)


def rand_distortion_2d(pred_arr):
    if random.uniform(0, 1) > .5:
        for _ in range(random.randint(0, 10)):
            selem_func = random.choice((disk, square, diamond, star))
            selem = selem_func(random.randint(3, 10))
            y_start = random.randint(
                10, pred_arr.shape[0] - 10 - selem.shape[0])
            x_start = random.randint(
                10, pred_arr.shape[1] - 10 - selem.shape[1])
            pred_arr[y_start:y_start+selem.shape[0],
                     x_start:x_start+selem.shape[1]] ^= selem

    selem = disk(3)
    for _ in range(3):
        if random.uniform(0, 1) > .5:
            pred_arr = dilation(pred_arr, selem)
        if random.uniform(0, 1) < .5:
            pred_arr = erosion(pred_arr, selem)
    return pred_arr


def rand_distortion_3d(pred_arr):
    if random.uniform(0, 1) > .5:
        for _ in range(random.randint(0, 10)):
            selem_func = random.choice((cube, octahedron, ball))
            selem = selem_func(random.randint(3, 10))
            y_start = random.randint(
                10, pred_arr.shape[0] - 10 - selem.shape[0])
            x_start = random.randint(
                10, pred_arr.shape[1] - 10 - selem.shape[1])
            z_start = random.randint(
                10, pred_arr.shape[2] - 10 - selem.shape[2])
            pred_arr[y_start:y_start+selem.shape[0],
                     x_start:x_start+selem.shape[1],
                     z_start:z_start+selem.shape[2]] ^= selem

    selem = ball(3)
    for _ in range(3):
        if random.uniform(0, 1) > .5:
            pred_arr = dilation(pred_arr, selem)
        if random.uniform(0, 1) < .5:
            pred_arr = erosion(pred_arr, selem)
    return pred_arr


def nifity_to_array(path):
    itk_data = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(itk_data)
    return array


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
    return out.astype(np.float32)


def resize_ND_volume_to_given_shape(volume, out_shape, order=3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0 = volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order=order)
    return out_volume


def geodesic_distance_3d(I, S, spacing, lamb, iter):
    '''
    get 3d geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)


def remove_small_region(img, threshold=100):
    s = ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]

    out_img = np.zeros_like(img)
    for temp_size in sizes_list:
        if(temp_size > threshold):
            temp_lab = np.where(sizes == temp_size)[0] + 1
            temp_cmp = labeled_array == temp_lab
            out_img = (out_img + temp_cmp) > 0
    return out_img


def get_random_seeds(label, maxN):
    '''
    select a random number of non-zero pixels as seeds from a binary image
    '''
    seeds = np.zeros(label.shape, np.uint8)
    if(maxN > 0):
        seedn = max(random.randrange(0, maxN), 1)
        b = np.where(label > 0)
        seedLen = len(b[0])
        if(seedLen > seedn):
            seedlist = random.sample(range(seedLen), seedn)
            for i in range(seedn):
                seed_idx = seedlist[i]
                seeds[b[0][seed_idx]][b[1][seed_idx]][b[2][seed_idx]] = 1
    struc = ndimage.generate_binary_structure(3, 1)
    seeds = ndimage.binary_dilation(seeds, struc)
    return seeds


def interaction_geodesic_distance(img, seed, dis_threshold):
    print("seed sum", seed.sum())
    if(seed.sum() > 0):
        geo_dis = geodesic_distance_3d(img, seed, [1, 1, 1], 1.0, 2)
        geo_dis[geo_dis > dis_threshold] = dis_threshold
        dis = geo_dis/(geo_dis.max()+1e-8)  # recale to 0-1
    else:
        dis = np.zeros_like(img, np.float32)
    return dis


def interaction_euclidean_distance(seed):
    threshold = 4.0
    if(seed.sum() > 0):
        euc_dis = ndimage.distance_transform_edt(seed == 0)
        euc_dis[euc_dis > threshold] = threshold
        dis = euc_dis/threshold
    else:
        dis = np.zeros_like(seed, np.float32)
    euc_dis = cstm_normalize(dis, 1.0)
    return euc_dis


def gaussian_kernel(d, bias=0, sigma=10):
    """
    this a gaussian kernel
    input:
        d: distance between each extreme point to every point in volume
        bias:
        sigma: is full-width-half-maximum, which can be thought of as an effective radius.
    """
    gaus_dis = (1 / (sigma * np.sqrt(2 * np.pi))) * \
        np.exp(- ((d - bias)**2 / (2 * sigma**2)))
    return gaus_dis


def cstm_normalize(im, max_value):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value*(im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def resize_3D_volume_to_given_shape(volume, out_shape, order=3):
    shape0 = volume.shape
    scale_d = (out_shape[0]+0.0)/shape0[0]
    scale_h = (out_shape[1]+0.0)/shape0[1]
    scale_w = (out_shape[2]+0.0)/shape0[2]
    return ndimage.interpolation.zoom(volume, [scale_d, scale_h, scale_w], order=order)


def interaction_gaussian_distribution(img, seeds, bias=0, sigma=10):
    """
    the gaussian distribution of euclidean distance
    """
    if(seeds.sum() > 0):
        reshape_seeds = ndimage.zoom(seeds, 0.5, order=0)
        dis = ndimage.distance_transform_edt(reshape_seeds == 0)
        gaussian_dis = gaussian_kernel(dis, bias, sigma)
    else:
        dis = np.ones_like(img, np.float32)
        gaussian_dis = gaussian_kernel(dis, bias, sigma)
    gaussian_dis = cstm_normalize(gaussian_dis, 1.0)
    return gaussian_dis


def generate_interaction_distance_one_image(img, lab, seg, seed_org, transform_type=None,
                                            size_threshold=100, seed_sparsity=100, distance_threshold=0.4):
    seg = seg.astype(np.uint8)
    seg = rand_distortion_3d(seg)
    img_down = ndimage.zoom(img, 0.5, order=1)
    lab_down = ndimage.zoom(lab, 0.5, order=0)
    seg_down = ndimage.zoom(seg, 0.5, order=0)
    fore_seed_org_down = ndimage.zoom(seed_org[0], 0.5, order=0)
    back_seed_org_down = ndimage.zoom(seed_org[1], 0.5, order=0)

    fg = lab_down * (1.0 - seg_down)
    # fg = remove_small_region(fg, size_threshold)
    print("fg.sum()", fg.sum())
    maxN = int(fg.sum()/seed_sparsity)
    fg_seeds = get_random_seeds(fg, maxN)
    if transform_type == "geodesic":
        fg_dis = interaction_geodesic_distance(
            img_down, fg_seeds.astype(np.uint8), distance_threshold)
    elif transform_type == "euclidean":
        fg_dis = interaction_euclidean_distance(fg_seeds)

    elif transform_type == "gaussian":
        fg_dis = interaction_euclidean_distance(fg_seeds)

    bg = (1.0 - lab_down) * seg_down
    # bg = remove_small_region(bg, size_threshold)
    maxN = int(bg.sum()/seed_sparsity)
    bg_seeds = get_random_seeds(bg, maxN)
    print("bg.sum()", bg.sum())
    if transform_type == "geodesic":
        bg_dis = interaction_geodesic_distance(
            img_down, bg_seeds.astype(np.uint8), distance_threshold)
    elif transform_type == "euclidean":
        bg_dis = interaction_euclidean_distance(bg_seeds)
    elif transform_type == "gaussian":
        bg_dis = interaction_euclidean_distance(bg_seeds)

    fg_dis = resize_ND_volume_to_given_shape(fg_dis, img.shape, 1)
    bg_dis = resize_ND_volume_to_given_shape(bg_dis, img.shape, 1)
    fg_seeds = resize_ND_volume_to_given_shape(fg_seeds, img.shape, 0)
    bg_seeds = resize_ND_volume_to_given_shape(bg_seeds, img.shape, 0)
    return [fg_dis, bg_dis, fg_seeds, bg_seeds]


def get_3dimage_largest_component(img):
    s = ndimage.generate_binary_structure(3, 1)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches+1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    labeled_array == max_label
    centroid = measure.regionprops(labeled_array)[0]["centroid"]
    return labeled_array, centroid


def extends_points(seed):
    if(seed.sum() > 0):
        points = ndimage.distance_transform_edt(seed == 0)
        points[points > 1] = 2
        points[points < 2] = 1
        points[points >= 2] = 0
        return points.astype(np.uint8)


def generate_interaction_distance_one_image_validation(img, lab, seg, seed_org, transform_type=None,
                                                       size_threshold=50, seed_sparsity=100, distance_threshold=0):
    img_down = ndimage.zoom(img, 0.5, order=1)
    lab_down = ndimage.zoom(lab, 0.5, order=0)
    seg_down = ndimage.zoom(seg, 0.5, order=0)
    fore_seed_org_down = ndimage.zoom(seed_org[0], 0.5, order=0)
    back_seed_org_down = ndimage.zoom(seed_org[1], 0.5, order=0)

    fg = lab_down * (1.0 - seg_down)
    fg = remove_small_region(fg, size_threshold)
    print("fg", fg.sum())
    if fg.sum() < 100:
        fg_seeds = fore_seed_org_down
    else:
        largest_error_region, centroid = get_3dimage_largest_component(fg)
        fg_seeds = np.zeros_like(fg)
        fg_seeds[int(centroid[0]), int(centroid[1]), int(centroid[2])] = 1
        fg_seeds = extends_points(fg_seeds)
    if transform_type == "geodesic":
        fg_dis = interaction_geodesic_distance(
            img_down, fg_seeds.astype(np.uint8), distance_threshold)
    elif transform_type == "euclidean":
        fg_dis = interaction_euclidean_distance(fg_seeds)

    elif transform_type == "gaussian":
        fg_dis = interaction_euclidean_distance(fg_seeds)

    bg = (1.0 - lab_down) * seg_down
    bg = remove_small_region(bg, size_threshold)
    print("bg", bg.sum())
    if bg.sum() < 100:
        bg_seeds = back_seed_org_down
    else:
        largest_error_region, centroid = get_3dimage_largest_component(bg)
        bg_seeds = np.zeros_like(bg)
        bg_seeds[int(centroid[0]), int(centroid[1]), int(centroid[2])] = 1
        bg_seeds = extends_points(bg_seeds)

    if transform_type == "geodesic":
        bg_dis = interaction_geodesic_distance(
            img_down, bg_seeds.astype(np.uint8), distance_threshold)
    elif transform_type == "euclidean":
        bg_dis = interaction_euclidean_distance(bg_seeds)
    elif transform_type == "gaussian":
        bg_dis = interaction_euclidean_distance(bg_seeds)
    fg_dis = resize_ND_volume_to_given_shape(fg_dis, img.shape, 1)
    bg_dis = resize_ND_volume_to_given_shape(bg_dis, img.shape, 1)
    fg_seeds = resize_ND_volume_to_given_shape(fg_seeds, img.shape, 0)
    bg_seeds = resize_ND_volume_to_given_shape(bg_seeds, img.shape, 0)
    return [fg_dis, bg_dis, fg_seeds, bg_seeds]


def generate_simulation_deepigeos(img, lab, pred, seed_org, transform_type=None,
                                  size_threshold=200, seed_sparsity=300, distance_threshold=0.4):
    """
    img, lab, seg are tensor.
    """
    batch_size = img.shape[0]
    pred = torch.softmax(pred, dim=1)[:, 1, ...].unsqueeze(1)
    img = img.cpu().data.numpy()
    lab = lab.cpu().data.numpy()
    pred = pred.cpu().data.numpy()
    seed_org = seed_org.cpu().data.numpy()
    inputs = []
    geos_maps = []
    seeds = []
    for b in range(batch_size):
        fg_dis, bg_dis, fg_seeds, bg_seeds = generate_interaction_distance_one_image(
            img[b, 0, ...], lab[b, ...], pred[b, 0, ...] >= 0.5, seed_org[b, ...], transform_type, distance_threshold=0.4)
        inputs.append([img[b, 0, ...], (pred[b, 0, ...] >=
                      0.5).astype(np.uint8), fg_dis, bg_dis])
        geos_maps.append([fg_dis, bg_dis])
        seeds.append([fg_seeds, bg_seeds])
    inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).cuda()
    geos_maps = torch.from_numpy(np.array(geos_maps)).cuda()
    seeds = torch.from_numpy(np.array(seeds)).cuda()
    return [inputs, geos_maps, seeds]


def generate_simulation_deepigeos_validation(img, lab, pred, seed_org, transform_type=None,
                                             size_threshold=200, seed_sparsity=300, distance_threshold=0.4):
    """
    img, lab, seg are tensor.
    """
    batch_size = img.shape[0]
    pred = torch.softmax(pred, dim=1)[:, 1, ...].unsqueeze(1)
    img = img.cpu().data.numpy()
    lab = lab.cpu().data.numpy()
    pred = pred.cpu().data.numpy()
    seed_org = seed_org.cpu().data.numpy()
    inputs = []
    geos_maps = []
    seeds = []
    for b in range(batch_size):
        fg_dis, bg_dis, fg_seeds, bg_seeds = generate_interaction_distance_one_image_validation(
            img[b, 0, ...], lab[b, ...], pred[b, 0, ...] >= 0.5, seed_org[b, ...], transform_type, distance_threshold=0.4)
        inputs.append([img[b, 0, ...], (pred[b, 0, ...] >=
                      0.5).astype(np.uint8), fg_dis, bg_dis])
        geos_maps.append([fg_dis, bg_dis])
        seeds.append([fg_seeds, bg_seeds])
    inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).cuda()
    geos_maps = torch.from_numpy(np.array(geos_maps)).cuda()
    seeds = torch.from_numpy(np.array(seeds)).cuda()
    return [inputs, geos_maps, seeds]

# if __name__ == "__main__":
#     img_path = "/home/SENSETIME/luoxiangde/Projects/RInterSeg/data/BraTS2018/autoSeg/Brats18_TCIA02_322_1_img.nii.gz"
#     lab_path = "/home/SENSETIME/luoxiangde/Projects/RInterSeg/data/BraTS2018/autoSeg/Brats18_TCIA02_322_1_lab.nii.gz"
#     autoseg_path = "/home/SENSETIME/luoxiangde/Projects/RInterSeg/data/BraTS2018/autoSeg/Brats18_TCIA02_322_1_seg.nii.gz"
#     image = nifity_to_array(img_path)
#     image = itensity_normalize_one_volume(image)
#     label = nifity_to_array(lab_path)
#     autoseg = nifity_to_array(autoseg_path)
#     seed_org = np.zeros(
#         (2, autoseg.shape[0], autoseg.shape[1], autoseg.shape[2]))
#     fg_dis, bg_dis, fg_seeds, bg_seeds = generate_interaction_distance_one_image_validation(
#         image, label, autoseg, seed_org, transform_type="geodesic", distance_threshold=0.4)
#     sitk.WriteImage(sitk.GetImageFromArray(fg_dis), "fg_geo_dis1.nii.gz")
#     sitk.WriteImage(sitk.GetImageFromArray(bg_dis), "bg_geo_dis1.nii.gz")
#     sitk.WriteImage(sitk.GetImageFromArray(
#         fg_seeds.astype(np.float32)), "fg_seeds.nii.gz")
    # sitk.WriteImage(sitk.GetImageFromArray(
    #     bg_seeds.astype(np.float32)), "bg_seeds.nii.gz")
