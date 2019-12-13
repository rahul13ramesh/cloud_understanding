import numpy as np
import pandas as pd
import torch
import datetime
from torch.optim import Adam
import torch.nn as nn
from torchsummary import summary
from loss import dice, dice_pytorch, dice_BCE
from models.model_classifier import ConvModel
from data_fetch import get_data
from models.resnet_models import Wide_ResNet

torch.manual_seed(100)
num_points_fetch = -1
train_on_gpu = True
n_epochs = 50
batch_size = 4
train_num_pts = 4800
model_string = str(datetime.datetime.now())

#  model = ConvModel(dropval=0.5, l1=64, l2=128, l3=256)
model = Wide_ResNet(28, 10, 0.5)
model = model.cuda()
summary(model, (3, 140, 210))

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.0004)

print(model_string)
print("Reading data")
all_data, _, img_name, class_labels = get_data(num_points_fetch)
print("Loaded data")

seg_labels = pd.read_csv("~/downloads/data/train.csv")
num_total = len(seg_labels)//4

# Create labels
labels = []
for i in range(num_total):
    cur_lab = []
    for j in range(4):
        if type(seg_labels.iloc[4 * i + j, 1]) == float:
            cur_lab.append(0.0)
        else:
            cur_lab.append(1.0)
    labels.append(cur_lab)
labels = np.array(labels)

all_data = all_data / 255.0

test_data = all_data[train_num_pts:]
test_label = labels[train_num_pts:]

all_data = all_data[:train_num_pts]
labels = labels[:train_num_pts]
numpts = len(all_data)
indices = list(range(numpts))

for epoch in range(1, n_epochs+1):
    np.random.shuffle(indices)
    all_data = all_data[indices]
    labels = labels[indices]

    train_loss = 0.0
    err = 0.0
    numpts = len(test_data)

    model.eval()
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

        out = np.array(out[0] > 0.5, dtype=int)
        pred = np.array(target_pt[0], dtype=int)

        err += np.abs(out - pred)
        train_loss += loss.item()*data.size(0)

    err = err / numpts
    train_loss = train_loss / numpts
    print("----------")
    print("Test err: " + str(err))
    print("Test Loss: " + str(train_loss))

    model.train()
    train_loss = 0.0
    numpts = len(all_data)
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

    train_loss = train_loss / (numpts)
    print("Epoch " + str(epoch) + " loss = " + str(train_loss))

torch.save(model.state_dict(), "checkpoint/classifier-" + model_string + ".ckpt")
