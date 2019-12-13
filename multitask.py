import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torchsummary import summary
from loss import dice, dice_pytorch
from loss import DistToBoundary
from loss import boundary_distance_dice
from models.seg_classifier import UNet_classifier
from data_fetch import get_data


train_on_gpu = True
LOSS_NUM = 0
LR = 0.0004
WT_Classifier = 5
batch_size = 4
model_string = str(datetime.datetime.now())

#  Code testing config
#  num_points_fetch = 10
#  train_num_pts = 5
#  n_epochs = 4
train_num_pts = 4800
num_points_fetch = -1
n_epochs = 40

# Pick model
model = UNet_classifier(n_channels=3, n_classes=4).float()
model = model.cuda()
summary(model, (3, 140, 210))

seg_labels = pd.read_csv("~/downloads/data/train.csv")
num_total = len(seg_labels)//4


# Print parameter choices
print("Learning rate: " + str(LR))
print(model_string)
print("Classifier weightL: " + str(WT_Classifier))

class_crit = nn.BCELoss()
if LOSS_NUM == 0:
    print("Using Dice loss")
    criterion = dice_pytorch
elif LOSS_NUM == 1:
    print("Using distance weighted Dice")
    criterion = boundary_distance_dice


# Pick optimizer
optimizer = Adam(model.parameters(), lr=LR)

print("Reading data")
all_data, labels, img_name, class_labels = get_data(num_points_fetch)
all_data = all_data / 255.0

class_labels = []
for i in range(num_total):
    cur_lab = []
    for j in range(4):
        if type(seg_labels.iloc[4 * i + j, 1]) == float:
            cur_lab.append(0.0)
        else:
            cur_lab.append(1.0)
    class_labels.append(cur_lab)
class_labels = np.array(class_labels)
print("Loaded data")

if LOSS_NUM == 1:
    print("calculating weights")
    obj = DistToBoundary(labels)
    rowWts, colWts = obj.computeWeights()
    weights = 5 * np.exp(-(rowWts + colWts)/50.0)

# Split data into train and test
test_data = all_data[train_num_pts:]
test_label = labels[train_num_pts:]
test_class = class_labels[train_num_pts:]

all_data = all_data[:train_num_pts]
labels = labels[:train_num_pts]
class_labels = class_labels[:train_num_pts]
numpts = len(all_data)
indices = list(range(len(all_data)))

if LOSS_NUM == 1:
    test_weights = weights[train_num_pts:]
    weights = weights[:train_num_pts]

# Start training
print("Training model")
for epoch in range(1, n_epochs+1):

    model.eval()
    test_loss = 0.0
    dice_val = 0.0
    err = 0.0
    numpts = len(test_data)

    for i in range(numpts):
        data = test_data[i:i+1]
        target_pt = test_label[i:i+1]
        class_lab = class_labels[i:i+1]

        if LOSS_NUM == 1:
            test_wt = test_label[i:i+1]
            wt = torch.from_numpy(test_wt).float().cuda()

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target_pt).float()

        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output, class_output = model(data)

        if LOSS_NUM == 1:
            loss = criterion(output, target, wt)
        else:
            loss = criterion(output, target)

        class_output = class_output.cpu().detach().numpy()
        out = output.cpu().detach().numpy()

        test_loss += loss.item() * data.size(0)

        out = np.array(out[0] > 0.5)
        dice_val += dice(out, target_pt[0])

        out_lab = np.array(class_output[0] > 0.5, dtype=int)
        pred_lab = np.array(class_lab[0], dtype=int)
        err += np.abs(out_lab - pred_lab)

    dice_val = dice_val / numpts
    test_loss = test_loss / numpts
    print("Dice: " + str(dice_val))
    print("Loss: " + str(test_loss))
    print("Err: " + str(err))

    # Training
    model.train()
    np.random.shuffle(indices)
    all_data = all_data[indices]
    labels = labels[indices]
    class_labels = class_labels[indices]
    train_loss = 0.0
    numpts = len(all_data)

    for i in range(0, numpts, batch_size):
        data = all_data[i:i+batch_size]
        target = labels[i:i+batch_size]
        class_lab = class_labels[i:i+batch_size]

        if LOSS_NUM == 1:
            wts = weights[i:i+batch_size]
            wts = torch.from_numpy(wts).float().cuda()

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()
        class_lab = torch.from_numpy(class_lab).float()

        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
            class_lab = class_lab.cuda()

        optimizer.zero_grad()
        output, class_output = model(data)

        if LOSS_NUM == 1:
            loss = criterion(output, target, wts) +\
                (0.2 * class_crit(class_output, class_lab))
        else:
            loss = criterion(output, target) + \
                (0.2 * class_crit(class_output, class_lab))

        loss.backward()

        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss = train_loss / numpts
    print("Epoch " + str(epoch) + " loss = " + str(train_loss))
    print("-------------")

torch.save(model.state_dict(), "checkpoint/multi-" + model_string + ".ckpt")
