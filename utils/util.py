import csv
import importlib
import os

import cv2
import cv2 as cv
import numpy as np
import torch
from natsort import natsorted
from torchvision import transforms
from torchvision.utils import save_image
import albumentations as A

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def read_cube_to_np(img_dir, stack_axis=2, cvflag=cv.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"got {img_dir}"
    print(img_dir)
    imgs = []
    names = natsorted(os.listdir(img_dir))
    for name in names:
        img = cv.imread(os.path.join(img_dir, name), cvflag)
        imgs.append(img)
    imgs = np.stack(imgs, axis=stack_axis)
    return imgs


def read_cube_to_tensor(path, stack_axis=1, cvflag=cv.IMREAD_GRAYSCALE):
    imgs = []
    names = natsorted(os.listdir(path))
    for name in names:
        img = cv.imread(os.path.join(path, name), cvflag)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=stack_axis)
    return imgs

def _array_path_evaluation_recursively(tsv_path=None, iou_threshold=1., only_return_paths=False):
    msgs = []
    name = 1
    if tsv_path is not None:
        with open(tsv_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                path, dice_score, dsc_score, iou_score, fnr_score, fpr_score, h95_score = row
                if name > 0:
                    name -= 1
                    continue

                if float(iou_score)>iou_threshold:
                    continue
                if only_return_paths:
                    path = path.replace('img','gt')
                    msgs.append(path)
                else:
                    msgs.append([path, dice_score, dsc_score, iou_score, fnr_score, fpr_score, h95_score])

    return msgs



def _array_image_paths_recursively(tsv_path=None, sub=False, max_len=6000,is_test=False):
    images_paths = []
    if tsv_path is not None:
        with open(tsv_path) as csvfile:
            reader = csv.reader(csvfile,delimiter='\t')
            for row in reader:
                paths = row[0]
                images_paths.append(paths)
    if sub:
        sub_r = len(images_paths) // 10 * 9
        sub_r = min(max_len, sub_r)
        if is_test:
            end_r = min(sub_r+max_len, len(images_paths))
            images_paths = images_paths[sub_r:end_r]
        else:
            images_paths = images_paths[:sub_r]

    return images_paths

def select_ped_label(mask_croped, opt_del=True, path=None):
    class_label = 0
    if np.sum(mask_croped[mask_croped>0]) > 0:
        class_label = 1
    res = np.zeros(mask_croped.shape)
    if '100edema' in path:
        res[mask_croped==128] = 255
    else:
        res[mask_croped==255] = 255
    return res, class_label

def select_srf_label(mask_croped, opt_del=True):
    class_label = 0
    if np.sum(mask_croped[mask_croped>0]) > 0:
        class_label = 1
    if opt_del:
        mask_croped[mask_croped!=191] = 0 # del 2017challenge PED 255 / 100edema REA
        mask_croped[mask_croped==191] = 255
    return mask_croped, class_label


def select_labels(mask_croped):
    assert(np.sum(mask_croped[mask_croped == 50]) == 0)
    assert (np.sum(mask_croped[mask_croped == 150]) == 0)

    #  srf 1, ped 3 #
    class_label = [0, 0]  # [0, 0]
    if np.sum(mask_croped[mask_croped == 191]) > 0:
        class_label[0] = 1
    if np.sum(mask_croped[mask_croped == 128]) > 0:
        class_label[1] = 1
    mask_croped[mask_croped == 255 ] = 0
    mask_croped[ mask_croped == 191 ] = 255

    return mask_croped, class_label[0]*2+class_label[1]


def trans(self, pil_image, load_size=256, hor_p=0.):
    ts = A.Compose(
        [
            A.Resize(load_size, load_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            A.HorizontalFlip(p=hor_p),
        ]
    )
    pil_image = ts(image=pil_image)['image']
    return pil_image

def cut_to_croped(image, mask=None, layer=None, img_path=None, th=39,mit_h=640):
    img = image
    try:
        blured = cv2.medianBlur(img, th)
    except:
        print(img_path)
    area = np.where(blured > th+1)
    # if len(area[0])==0:
    #     print("ok")
    #     print("img_path")
    dl = 0
    limit_h = max(area[0]) - min(area[0])
    if limit_h < mit_h:
        dl = (mit_h - limit_h) //2 + 1
    h_l = max( min(area[0])-dl, 0)
    h_r = min( max(area[0])+dl, img.shape[0])

    img_croped = img[h_l:h_r]
    mask_croped = mask
    layer_croped = mask
    if mask is not None:
        mask_croped = mask[h_l:h_r]
    if layer is not None:
        layer_croped = layer[h_l:h_r]
    return {'image':img_croped,"mask": mask_croped,"layer":layer_croped}

def save_cube_from_tensor(img, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    for j in range(img.shape[0]):
        img_path = os.path.join(result_dir, str(j + 1) + '.png')
        save_image(img[j, :, :], img_path)


def save_cube_from_numpy(data, result_name, tonpy=False):
    if tonpy:
        np.save(result_name + '.npy', data)
    else:
        result_dir = result_name
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        for i in range(data.shape[0]):
            cv.imwrite(os.path.join(result_dir, str(i + 1) + '.png'), data[i, ...])


def get_file_path_from_dir(src_dir, file_name):
    for root, dirs, files in os.walk(src_dir):
        if file_name in files:
            print(root, dirs, files)
            return os.path.join(root, file_name)