import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from torchsummary import summary
from loss import dice, dice_pytorch, dice_BCE
from models.model_classifier import ConvModel
from data_fetch import get_data

num_points_fetch = -1
train_on_gpu = True
model = ConvModel()
model = model.cuda()
summary(model, (3, 140, 210))

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.005)

print("Reading data")
all_data, labels, img_name, class_labels = get_data(num_points_fetch)
print("Loaded data")
all_data = all_data / 255.0

label_vals = []
labels = np.sum(labels, axis=(2, 3))
labels = np.array(labels > 0.5, dtype=np.float32)

n_epochs = 20
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
accuracy = 0.0
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

    out = np.array(out[0] > 0.5, dtype=int)
    pred = np.array(target_pt[0], dtype=int)

    accuracy += np.sum(np.abs(out - pred))
    train_loss += loss.item()*data.size(0)

accuracy = accuracy / (4 * numpts)
train_loss = train_loss / (4 * numpts)
print("Acc: " + str(accuracy))
print("Loss: " + str(train_loss))
