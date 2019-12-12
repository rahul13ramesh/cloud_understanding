import numpy as np
import datetime
import matplotlib 

import torch
from torch.optim import Adam
from torchsummary import summary
import torch.nn as nn
from loss import dice, dice_pytorch, dice_BCE
from model import UNet
from data_fetch import get_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.use("Agg")


LOSS_NUM = 1
batch_size = 4
model_string = "checkpoint/model-2019-12-11 16:12:41.597676.ckpt"

# Code testing config
#  num_points_fetch = 30
#  train_num_pts = 10
#  n_epochs = 1

train_num_pts = 4800
num_points_fetch = -1
n_epochs = 80

# Pick model
train_on_gpu = True
model = UNet(n_channels=3, n_classes=4).float()
model = model.cuda()

model.load_state_dict(torch.load(model_string))
model.eval()

#  summary(model, (3, 140, 210))

print("Reading data")
all_data, labels, img_name, class_labels = get_data(num_points_fetch)
all_data = all_data / 255.0
print("Loaded data")

# Split data into train and test

test_data = all_data[train_num_pts:]
test_label = labels[train_num_pts:]

all_data = all_data[:train_num_pts]
labels = labels[:train_num_pts]
indices = list(range(len(all_data)))

# Start training
print("Evaluating model")
dice_val = 0.0
numpts = len(test_data)

for i in range(numpts):
    data = test_data[i:i+1]
    target_pt = test_label[i:i+1]

    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target_pt).float()

    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    output = model(data)
    out = output.cpu().detach().numpy()
    out = np.array(out[0] > 0.5)

    dice_val += dice(out, target_pt[0])


for i in range(30):
    data = test_data[i:i+1]
    target_pt = test_label[i:i+1]
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target_pt).float()

    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    output = model(data)
    out = output.cpu().detach().numpy()
    #  out = np.array(out[0] > 0.5)

    plt.figure(figsize=(15, 10))
    f, ax = plt.subplots(2, 4)
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.025, hspace=0.05)

    for j in range(4):
        ax[0, j].imshow(out[0][j], cmap="gray")
        ax[0, j].axis('off')
    for j in range(4):
        ax[1, j].imshow(target_pt[0][j], cmap="gray")
        ax[1, j].axis('off')
    plt.savefig("plots/true-" + str(i) + ".png")
    plt.close()


