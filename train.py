import numpy as np
import torch
from torch.optim import Adam
from torchsummary import summary
import torch.nn as nn
from loss import dice, dice_pytorch, dice_BCE
from model import UNet
from data_fetch import get_data

LOSS_NUM = 2

num_points_fetch = -1
train_on_gpu = True
model = UNet(n_channels=3, n_classes=4).float()
model = model.cuda()

summary(model, (1, 140, 210))

print("Loss is " + str(LOSS_NUM))
if LOSS_NUM == 0:
    criterion = nn.BCELoss()
elif LOSS_NUM == 1:
    criterion = dice_pytorch
elif LOSS_NUM == 2:
    criterion = dice_BCE

optimizer = Adam(model.parameters(), lr=0.005)

print("Reading data")
all_data, labels, img_name, class_labels = get_data(num_points_fetch)
print("Loaded data")

n_epochs = 30
batch_size = 4
indices = list(range(len(all_data)))
model.train()

print(all_data.shape)
train_num_pts = 4800
test_data = all_data[train_num_pts:]
test_label = labels[train_num_pts:]

all_data = all_data[:train_num_pts]
labels = labels[:train_num_pts]

numpts = len(all_data)

for epoch in range(1, n_epochs+1):
    np.random.shuffle(indices)
    train_loss = 0.0

    for i in range(0, numpts, batch_size):
        data = all_data[i:i+batch_size]
        target = labels[i:i+batch_size]
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()

        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()*data.size(0)

    train_loss = train_loss / numpts
    print("Epoch " + str(epoch) + " loss = " + str(train_loss))

model.eval()
train_loss = 0.0
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
    loss = criterion(output, target)
    out = output.cpu().detach().numpy()
    out = np.array(out[0] > 0.3)

    dice_val += dice(out, target_pt[0])
    train_loss += loss.item()*data.size(0)

dice_val = dice_val / numpts
train_loss = train_loss / numpts
print("Dice: " + str(dice_val))
print("Loss: " + str(train_loss))
