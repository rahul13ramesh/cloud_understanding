import numpy as np
import datetime
import torch
from torch.optim import Adam
from torchsummary import summary
import torch.nn as nn
from loss import dice, dice_pytorch, dice_BCE
from loss import DistToBoundary
from loss import boundary_distance_BCE
from loss import boundary_distance_dice
from model import UNet
from data_fetch import get_data

LOSS_NUM = 4
LR = 0.0004
AUG = 2
batch_size = 4
model_string = str(datetime.datetime.now())

# Code testing config
#  num_points_fetch = 30
#  train_num_pts = 10
#  n_epochs = 1

train_num_pts = 4800
num_points_fetch = -1
n_epochs = 40

# Pick model
train_on_gpu = True
model = UNet(n_channels=3, n_classes=4).float()
model = model.cuda()
summary(model, (3, 140, 210))

# Print parameter choices
print("Learning rate: " + str(LR))
print("Augmentation: " + str(AUG))

# Pick loss
if LOSS_NUM == 0:
    print("Using Binary cross entropy")
    criterion = nn.BCELoss()
elif LOSS_NUM == 1:
    print("Using Dice loss")
    criterion = dice_pytorch
elif LOSS_NUM == 2:
    print("Using Combination of Dice and BCE")
    criterion = dice_BCE
elif LOSS_NUM == 3:
    print("Using distance weighted BCE")
    criterion = boundary_distance_BCE
elif LOSS_NUM == 4:
    print("Using distance weighted Dice")
    criterion = boundary_distance_dice


# Pick optimizer
optimizer = Adam(model.parameters(), lr=LR)

print("Reading data")
all_data, labels, img_name, class_labels = get_data(num_points_fetch)
all_data = all_data / 255.0
print("Loaded data")

if LOSS_NUM == 3 or LOSS_NUM == 4:
    print("calculating weights")
    obj = DistToBoundary(labels)
    rowWts, colWts = obj.computeWeights()
    weights = 5 * np.exp(-(rowWts + colWts)/50.0)

# Split data into train and test

test_data = all_data[train_num_pts:]
test_label = labels[train_num_pts:]

all_data = all_data[:train_num_pts]
labels = labels[:train_num_pts]
numpts = len(all_data)
indices = list(range(len(all_data)))

if LOSS_NUM == 3 or LOSS_NUM == 4:
    test_weights = weights[train_num_pts:]
    weights = weights[:train_num_pts]

# Start training
print("Training model")
for epoch in range(1, n_epochs+1):

    model.eval()
    test_loss = 0.0
    dice_val = 0.0
    numpts = len(test_data)

    for i in range(numpts):
        data = test_data[i:i+1]
        target_pt = test_label[i:i+1]

        if LOSS_NUM == 3 or LOSS_NUM == 4:
            test_wt = test_label[i:i+1]
            wt = torch.from_numpy(test_wt).float().cuda()

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target_pt).float()

        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        if LOSS_NUM == 3 or LOSS_NUM == 4:
            loss = criterion(output, target, wt)
        else:
            loss = criterion(output, target)

        out = output.cpu().detach().numpy()
        out = np.array(out[0] > 0.5)

        dice_val += dice(out, target_pt[0])
        test_loss += loss.item() * data.size(0)

    dice_val = dice_val / numpts
    test_loss = test_loss / numpts
    print("Dice: " + str(dice_val))
    print("Loss: " + str(test_loss))

    # Training
    model.train()
    np.random.shuffle(indices)
    all_data = all_data[indices]
    labels = labels[indices]
    train_loss = 0.0
    numpts = len(all_data)

    for i in range(0, numpts, batch_size):
        data = all_data[i:i+batch_size]
        target = labels[i:i+batch_size]

        if LOSS_NUM == 3 or LOSS_NUM == 4:
            wts = weights[i:i+batch_size]

        # Doesn't improve performance
        if AUG > 0:
            #  All augmentations:
            if AUG == 1:
                prob = np.random.rand()
                if prob < 0.2:
                    data = np.flip(data, 2).copy()
                    target = np.flip(target, 2).copy()
                    if LOSS_NUM == 3 or LOSS_NUM == 4:
                        wts = np.flip(target, 2).copy()
                elif prob < 0.4:
                    data = np.flip(data, 3).copy()
                    target = np.flip(target, 3).copy()
                    if LOSS_NUM == 3 or LOSS_NUM == 4:
                        wts = np.flip(target, 3).copy()
                elif prob < 0.6:
                    data = np.flip(data, (2, 3)).copy()
                    target = np.flip(target, (2, 3)).copy()
                    if LOSS_NUM == 3 or LOSS_NUM == 4:
                        wts = np.flip(target, (2, 3)).copy()
            #  All only horizontal flips
            if AUG == 2:
                prob = np.random.rand()
                if prob < 0.4:
                    data = np.flip(data, 2).copy()
                    target = np.flip(target, 2).copy()
                    if LOSS_NUM == 3 or LOSS_NUM == 4:
                        wts = np.flip(target, 2).copy()

        if LOSS_NUM == 3 or LOSS_NUM == 4:
            wts = torch.from_numpy(wts).float().cuda()

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()

        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        if LOSS_NUM == 3 or LOSS_NUM == 4:
            loss = criterion(output, target, wts)
        else:
            loss = criterion(output, target)

        loss.backward()

        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss = train_loss / numpts
    print("Epoch " + str(epoch) + " loss = " + str(train_loss))
    print("-------------")

torch.save(model.state_dict(), "checkpoint/model-" + model_string + ".ckpt")
