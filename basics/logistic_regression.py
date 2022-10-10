# imports 
import os 
import sys
from sklearn.utils import shuffle
import tqdm
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor

print(f'tqdm version is  :{tqdm.__version__}')
print(f'torch version is : {torch.__version__}')
print(f'numpy version is : {np.__version__}')

# download the dataste 
train_dataset = datasets.FashionMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=False
)

test_dataset = datasets.FashionMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=False
)

# Define data loading pipeline and defining parameters
epochs = 5
batch_size = 64
num_classes = 10
input_size = 28 * 28 
learning_rate = 0.01

# Define train loader and test loader 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# define the model 
model = nn.Linear(in_features=input_size,
                    out_features=num_classes)
# define the loss
loss = nn.CrossEntropyLoss() # for multi class problems 
# deifne optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Define the training loop
total_step = len(train_loader)
print(f'total steps are : {total_step}')
for epoch in tqdm.tqdm(range(epochs)):
    
    for i, (images, labels) in enumerate(train_loader):
        
        # Let's deifne the forward pass
        images = images.reshape(-1, input_size)
        outputs = model(images)
        
        # It is important that we place the (outpyts, labels) in right format for crossEntropy to work properly
        l = loss(outputs, labels)
        
        # Let's define the backward propagation and optimization calculation 
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    # Show loos at every iteration 
    print(f'loss is {l.item()}')

# Test the model 
# So now that training has been done, we need to make predictions and visualize the results
# since no learning is being done in testing, we need to turn off the gradient calculationwith torch.no_grad():
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


