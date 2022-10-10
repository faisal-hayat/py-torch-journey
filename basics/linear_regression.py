# import libraries
import os
import sys
import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

print(f'numpy version is  :{np.__version__}')
print(f'pandas version is  :{pd.__version__}')
print(f'torch version is : {torch.__version__}')


def main():
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    # Define the model 
    epochs = 50
    input_size = 1
    output_size = 1
    learning_rate = 0.01
    model = nn.Linear(in_features=input_size,
                      out_features=output_size)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)
    # Let's define the training loop
    for epoch in tqdm.tqdm(range(epochs)):
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)
        # Let's define the forward path 
        outputs = model(inputs)
        l = loss(targets, outputs)
        # Let's define the backward loop and optimizer
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f'loss is  :{l.item()}')

    # Let's display the model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


if __name__ == "__main__":
    main()