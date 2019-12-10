
"""
Read data and create data structures for visualization/analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = "/home/rahulram/downloads/data/"


def modifyImg(image_num, seg_labels):
    resizeshape = (210, 140)
    pixels = seg_labels.iloc[image_num, 1]
    name = seg_labels.iloc[image_num, 0].split("_")[0]
    label = seg_labels.iloc[image_num, 0].split("_")[1]

    img = cv2.imread(PATH + "train/" + name)
    shape = img.shape
    mask = np.zeros((shape[0] * shape[1]))

    if type(pixels) != float:
        pixels = pixels.split()
        pixels = [(int(pixels[i]),
                   int(pixels[i+1])) for i in range(0, len(pixels), 2)]

        for start, length in pixels:
            mask[start:start+length] = 1

    mask = mask.reshape(shape[0:2], order='F')
    img = cv2.resize(img, resizeshape)
    mask = cv2.resize(mask, resizeshape)

    name2 = seg_labels.iloc[image_num, 0].split(".")[0]
    cv2.imwrite(PATH + "train/" + name2 + "_small.png", img)
    cv2.imwrite(PATH + "train/" + name2 + "_mask_" + label + ".png", mask)


def main():
    seg_labels = pd.read_csv(PATH + "train.csv")
    num_pts = int(len(seg_labels)/4)

    for i in range(num_pts):
        if i % 100 == 0:
            print("Finished processing " + str(4*i) + " points")
        for j in range(4):
            modifyImg(4 * i + j, seg_labels)


if __name__ == '__main__':
    main()

