import numpy as np
import skimage


def iou(mask1, mask2):
    # 计算掩码的交集和并集
    intersection = np.sum((mask1 == 1) & (mask2 == 1))
    union = np.sum((mask1 == 1) | (mask2 == 1))
    # 计算IOU
    iou_score = intersection / union
    return iou_score

def hausdorff95(mask1, mask2):
    if np.sum(mask1) == 0 or np.sum(mask2) == 0:
        return 0.0

        # 计算掩码1到掩码2的距离
    dist1 = skimage.metrics.hausdorff_distance(mask1, mask2)
    hd1 = np.percentile(dist1, 95)

    # 计算掩码2到掩码1的距离
    dist2 = skimage.metrics.hausdorff_distance(mask2, mask1)
    hd2 = np.percentile(dist2, 95)

    # 取两个距离的最大值作为Hausdorff距离
    hausdorff_dist = max(hd1, hd2)

    return hausdorff_dist

def fnr(pred_mask, true_mask):
    # 计算实际为正样本但被错误地分类为负样本的样本数
    false_neg = np.sum((pred_mask == 0) & (true_mask == 1))

    # 计算实际为正样本的样本数
    true_pos = np.sum(true_mask == 1)
    area = np.sum((pred_mask == 1) | (true_mask == 1))
    # 计算FNR
    fnr = false_neg / area

    return fnr

def fpr(pred_mask, true_mask):
    # 计算实际为负样本但被错误地分类为正样本的样本数
    false_pos = np.sum((pred_mask == 1) & (true_mask == 0))

    area = np.sum((pred_mask == 1) | (true_mask == 1))


    # 计算实际为负样本的样本数
    true_neg = np.sum(true_mask == 0)

    # 计算FPR
    fpr = false_pos / area

    return fpr


def dsc(mask1, mask2):
    # 计算掩码的交集和并集
    intersection = np.sum((mask1 == 1) & (mask2 == 1))
    # 计算DSC
    dsc_score = (2 * intersection) / (np.sum(mask1) + np.sum(mask2))
    return dsc_score
