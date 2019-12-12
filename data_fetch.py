"""
Read data and create data structures for visualization/analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

#  PATH = "/home/rahulram/downloads/data/"
PATH = "../data/"


def getImgMask(image_num, seg_labels):
    name = seg_labels.iloc[image_num, 0].split(".")[0]
    class_lab = seg_labels.iloc[image_num, 0].split("_")[1]
    file_name = name + "_small.png"
    mask_name = name + "_mask_" + class_lab + ".png"
    img = cv2.imread(PATH + "train/" + file_name)
    mask = cv2.imread(PATH + "train/" + mask_name, 0)

    return img, mask, name, class_lab


def get_data(num_pts_fetch):
    seg_labels = pd.read_csv(PATH + "train.csv")
    max_len = int(len(seg_labels)/4)
    if num_pts_fetch <= 0:
        num_pts = max_len
    else:
        num_pts = num_pts_fetch
    data, labels = [], []
    class_lab = []
    img_name = []

    for i in range(num_pts):
        masks = []
        class_lab.append([])
        for j in range(4):
            img, mask, imgname, classname = getImgMask(4 * i + j, seg_labels)
            masks.append(np.array(mask))
            class_lab[i].append(classname)
        img_name.append(imgname)
        data.append(img)
        labels.append(masks)

    data = np.array(data)
    labels = np.array(labels)
    data = np.transpose(data, [0, 3, 1, 2])

    return data, labels, img_name, class_lab


if __name__ == '__main__':
    data, labels, img_name, class_labels = get_data(2)

