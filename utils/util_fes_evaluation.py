import csv
import os

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import label, regionprops


def bbox_msg_prop(props):
    # 获取能够包裹所有连通块的最小矩形
    min_row = min(prop.bbox[0] for prop in props)
    min_col = min(prop.bbox[1] for prop in props)
    max_row = max(prop.bbox[2] for prop in props)
    max_col = max(prop.bbox[3] for prop in props)

    # bounding_box = (min_row, min_col, max_row, max_col)
    return min_row, min_col, max_row, max_col

def min_rec_func(
        path=r"/home/Data/xiebaoye/100edema/gt/PDR0248_Macular Cube 512x128_5-22-2014_17-29-34_OS_sn22698_cube_z/67.bmp",
        only_label = False,
):

    # 加载掩码图像并转换为二值图像
    # mask_image = cv2.imread(r"D:\data\100edema_raw\gt\PDR0248_Macular Cube 512x128_5-22-2014_17-29-34_OS_sn22698_cube_z\63.bmp", 0)
    mask_image = cv2.imread(path, 0)
    # plt.imshow(mask_image,cmap='gray')
    # plt.show()

    # imag = cv2.imread(path.replace('gt','img'), 0)

    if not only_label:
        mask_image[mask_image == 255] = 0

    ret, binary_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

    # 使用label函数获取连通区域
    labels = label(binary_image)

    # 计算每个连通区域的regionprops
    props = regionprops(labels)
    #return len(props)
    # plt.imshow(imag)
    # plt.show()
    # plt.imshow(mask_image)
    # plt.show()
    # 遍历每个连通区域的regionprops

    area_angles = []
    center_arr = []
    sum_perimeter = 0.
    area_total = 0.
    convex_area_total = 0.
    for prop in props:
        # 获取最小外接矩形的属性
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        height = maxr - minr
        orientation = prop.orientation
        img = np.zeros((mask_image.shape[0], mask_image.shape[1], 3))
        # 调整矩形的旋转角度
        center_x = (minc + maxc) // 2
        center_y = (minr + maxr) // 2
        #center_arr.append( (center_x , center_y))
        # 将弧度转换为角度
        rotation_angle_degrees = np.degrees(orientation)

        area_angles.append( (prop.area, rotation_angle_degrees) )

        # rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle_degrees, 1.0)
        # rotated_image = cv2.warpAffine(mask_image, rotation_matrix, (mask_image.shape[1], mask_image.shape[0]))
        R = math.sqrt(prop.area / math.pi)
        area_total += prop.equivalent_diameter
        sum_perimeter += prop.perimeter
        convex_area_total += prop.convex_area
        #cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 255, 0), 2)

        # rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        # 打印矩形信息
        # print("最小外接矩形的左上角坐标：({}, {})".format(minc, minr))
        # print("最小外接矩形的宽度：{}".format(width))
        # print("最小外接矩形的高度：{}".format(height))
        # print("最小外接矩形的旋转角度：{}".format(rotation_angle_degrees))
        # print('连通块面积的等效圆直径:{}'.format(prop.equivalent_diameter))
        # print("R:", 2 * R)
        # print('质心坐标:{}'.format(prop.centroid))

        # plt.imshow(img)
        # plt.show()

    # 所有掩码区域的最小外接矩阵 （为了计算旋转中心点）
    min_row, min_col, max_row, max_col = bbox_msg_prop(props)
    bounding_box = (min_row, min_col, max_row, max_col)

    #cv2.rectangle(img, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
    #print("Bounding box coordinates:", bounding_box)
    # plt.imshow(img)
    # plt.show()
    sorted_list = sorted(area_angles, key=lambda x: x[0], reverse=True)
    rotation_angle_degrees = 90 - sorted_list[0][1]
    #print(rotation_angle_degrees)
    center_x = (min_col + max_col) // 2
    center_y = (min_row + max_row) // 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle_degrees, 1.0)
    rotated_image = cv2.warpAffine(mask_image, rotation_matrix, (mask_image.shape[1], mask_image.shape[0]))

    # 使用label函数获取连通区域
    rotated_labels = label(rotated_image)

    # 计算每个连通区域的regionprops
    rotated_props = regionprops(rotated_labels)
    min_row, min_col, max_row, max_col = bbox_msg_prop(rotated_props)
    bounding_box = (min_row, min_col, max_row, max_col)
    cv2.rectangle(img, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)

    # 通过逆操作获取近似还原的变换矩阵
    inverse_rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -rotation_angle_degrees, 1.0)
    approx_original_image = cv2.warpAffine(rotated_image, inverse_rotation_matrix,
                                           (rotated_image.shape[1], rotated_image.shape[0]))

    plt.imshow(img)
    plt.show()
    plt.imshow(mask_image)
    plt.show()
    plt.imshow(approx_original_image)
    plt.show()

    # 联通块个数  最小外接矩阵的高  最小外接矩阵的长  连通块的周长 旋转角度
    return {
        'prop_number': len(props),
         'rect_height': max_row - min_row,
        'rect_width': max_col - min_col,
         'perimeter': sum_perimeter,
        'rotation_angle': rotation_angle_degrees,
        'area_equivalent_diameter': area_total,
        'solidity': area_total/convex_area_total,
        'img_path': path,
    }


def Layer_to_Image(layers):
    assert (layers.shape[2] == 512 ) #(1, 1, 512)
    layers = layers[0]
    layers = layers * 0.5 + 0.5
    layers = (layers * 1024).int()
    img = np.zeros((1024, 512))
    for layer in layers:
        for x, y in enumerate(layer):
            if y >=1023:
                y = 1023
            if y<0 and y>=1024:
                print(y)
                assert(y>=0 and y<1024)
            img[y][x] = 1.
    return torch.tensor(img)
# 显示绘制了矩形的掩码图像


def layer_to_data(layers):
    assert (layers.shape[2] == 512)  # (1, 1, 512)
    layers = layers[0]
    layers = layers * 0.5 + 0.5
    layers = (layers * 1024).int()
    return layers


def _array_image_paths_recursively(tsv_path=None):
    images_paths = []

    if tsv_path is not None:
        with open(tsv_path) as csvfile:
            reader = csv.reader(csvfile,delimiter='\t')
            for row in reader:
                paths = row[0]
                images_paths.append(paths)
    return images_paths


def save_tsv(images_paths, tsv_name):

    if os.path.exists(tsv_name):
        os.remove(tsv_name)
    with open(tsv_name, 'a', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(['img_path', 'prop_number', 'rect_height', 'rect_width', 'perimeter', 'rotation_angle', 'area_equivalent_diameter', 'solidity'])
        for item in images_paths:
            #print(item[0], "----- ",item[1])
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow([item['img_path'],
                                 item['prop_number'], item['rect_height'], item['rect_width'],
                                 item['perimeter'], item['rotation_angle'], item['area_equivalent_diameter'],
                                 item['solidity'],
                                 ])


if __name__ == "__main__":

    b = [[1,2,3,4]]
    min_rec_func()
    exit(0)
    divide_dir = 'seven'
    divide_ratio = 0.7
    train_paths = f"/home/Data/xiebaoye/100edema/description_tsv/{divide_dir}/OnlySRF_{divide_ratio}_train.tsv"
    train_h_paths = f"/home/Data/xiebaoye/100edema/description_tsv/{divide_dir}/OnlyHealth_{divide_ratio}_train.tsv"
    test_paths = f"/home/Data/xiebaoye/100edema/description_tsv/{divide_dir}/OnlySRF_{round(1 - divide_ratio, 1)}_test.tsv"

    o_label = False


    # img_paths = _array_image_paths_recursively(train_h_paths)
    # for i in range(len(img_paths)):
    #     img_paths[i] = img_paths[i].replace('100edema', 'layer_srf_generator')
    #     assert(os.path.exists(img_paths[i]))
    # o_label = True

    #img_paths = _array_image_paths_recursively(train_paths)

    img_paths = _array_image_paths_recursively(test_paths)

    #msg_mask_rect_tsv = f"/home/Data/xiebaoye/100edema/description_tsv/{divide_dir}/OnlySRF_{divide_ratio}_h_train_rect.tsv"
    #msg_mask_rect_tsv = f"/home/Data/xiebaoye/100edema/description_tsv/{divide_dir}/OnlySRF_{divide_ratio}_train_rect.tsv"
    msg_mask_rect_tsv = f"/home/Data/xiebaoye/100edema/description_tsv/{divide_dir}/OnlySRF_{round(1 - divide_ratio, 1)}_test_rect.tsv"

    #width_arr = [0 for i in range(550)]

    from tqdm import tqdm
    from matplotlib import pylab as plt

    msg_arr = []
    for i, item in enumerate(tqdm(img_paths)):
        numb_prop = min_rec_func(path=item,only_label=o_label)

        #width_arr[numb_prop['rect_width']] += 1
        msg_arr.append(numb_prop)
        break
    #save_tsv(msg_arr, msg_mask_rect_tsv)

    #plt.plot([i for i in range(550)], width_arr, label='Deformed bottom Line')
    #plt.show()

    #print(numb_prop)

"""
 0.9 train : numb prop
0: 0
1: 2867
2: 217
3: 12
4: 3
5: 3
6: 1
7: 0
8: 0
9: 0

"""
# cv2.imshow("Mask Image with Bounding Boxes", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
