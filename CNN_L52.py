


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms




# Define transformation for training and testing data.
# For training, use RandomCrop, RandomRotation and Normalization with mean 0.1307 variance0.3081
# For testing, just use Normalization.
# train_transform = ...
# test_transform


train_transforms = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081]),
    transforms.Lambda(lambda x: torch.unsqueeze(x, 1)),
    transforms.Lambda(lambda x: torch.nn.functional.pad(x, (2, 2, 2, 2))),
])


test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081]),
    transforms.Lambda(lambda x: torch.unsqueeze(x, 1)),
    transforms.Lambda(lambda x: torch.nn.functional.pad(x, (2, 2, 2, 2))),
])

# Load the training and testing datasets from Pytorch
# train_data = ...
# test_data = ...

train_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transforms,
)
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=test_transforms,
)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Print out first image and its correponding label index using matplotlib.
# ...
image, label = train_data[0]

plt.imshow(image.squeeze())
plt.show()

print(image.shape)
print(len(train_data))
print(len(test_data))

"""LeNet-5

Construct LeNet-5 as learning model for Fashion MNIST classification task.



Validation result:
    - Total params: 61,706
    - Trainable params: 61,706
    - Non-trainable params: 0



**References:**
- http://yann.lecun.com/exdb/lenet/

"""

from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, padding=0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output

# Decide whether you want to run your model on cpu or gpu.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net().to(device)
summary(net, (1, 28, 28))

"""### 2.4 LeNet-5 Model Training

Train LeNet-5 model with specific training strategy. **(20 Points)**


Set batch size to **64** for training.
Pick **SGD optimizer** with learning rate of **0.1**, momentum of **0.9**, and **nesterov=True**, for model training.
Pick **cross-entropy** loss function for optimization and evaluation metrics is set to **accuracy**.
Train the model with **10 epochs** 
"""

### Train with SGD optimizer with learning rate =0.1, regularizer=4e-5, momentum=0.9
model = Net()
train_load = DataLoader(
    datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    ),
    batch_size=64,
    shuffle=True,
)

test_load = DataLoader(
    datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    ),
    batch_size=64,
    shuffle=True,
)


optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
  running_loss = 0.0
  running_acc = 0.0

  for images, labels in train_load:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    running_acc +=torch.sum(torch.argmax(outputs, dim=1) == labels).item()

  epoch_loss = running_loss / len(train_load)
  epoch_acc = running_acc / len(train_data)

  test_loss = 0.0
  test_acc = 0.0

  for images, labels in test_load:
    outputs = model(images)
    loss = criterion(outputs, labels)
    test_loss +=loss.item()
    test_acc += torch.sum(torch.argmax(outputs, dim=1) ==labels).item()

  test_loss /= len(test_load)
  test_acc /= len(test_data)

### Print out the evaluation results, including test loss and test accuracy.
print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

