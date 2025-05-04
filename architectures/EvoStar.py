import io

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class EvoCNN(nn.Module):
#     def __init__(self):
#         super(EvoCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2))
#         self.dropout1 = nn.Dropout(p=0.25)
#         self.fc1 = nn.Linear(64 * 6 * 6, 128)
#         self.dropout2 = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.ReLU()(x)
#         x = self.conv2(x)
#         x = nn.ReLU()(x)
#         x = self.pool(x)
#         x = self.dropout1(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         x = nn.Softmax(dim=1)(x)
#         return x

class EvoCNN(nn.Module):
    def __init__(self):
        super(EvoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64 * 11 * 11, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x




import torchsummary
from torchviz import make_dot
from graphviz import render
from PIL import Image
model = EvoCNN()
torchsummary.summary(model, input_size=(3, 50, 50))

# x = torch.randn(1, 3, 32, 32)
# y = model(x)
# img = make_dot(y, params=dict(model.named_parameters()))
# dot = make_dot(y)
# dot_format = dot.pipe(format='png')
# image = Image.open(io.BytesIO(dot_format))
# image.save('model.png')