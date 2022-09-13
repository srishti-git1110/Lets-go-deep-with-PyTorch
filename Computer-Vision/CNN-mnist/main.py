import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn as nn
from torch.nn import functional as F
from model import CNN
from train import train
from evaluate import test

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True)            

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor(),
    download = True)


figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()



train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=1)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=1)

cnn = CNN()
loss_func = nn.CrossEntropyLoss() 
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   

n_epochs = 10
cnn.train()

train(n_epochs, cnn, train_loader)
test(cnn, test_loader)
