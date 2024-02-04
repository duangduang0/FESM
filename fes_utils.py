import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import comb

def extract_contours(image):
    # rgb->gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gaussian filter
    gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # binary exp-threshold=0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # find contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def twist_curve(x, y, noise_factor = 0.01):
    # 对弧线进行无规则扭曲
    # noise_factor = 0.01  # 扭曲程度，可根据需要调整

    distortion = noise_factor * np.random.randn(len(x))
    x_distorted = x + distortion
    y_distorted = y + distortion

    return x_distorted, y_distorted
