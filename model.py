import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


# Model to accept (1,10) input and output (1,10)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(10,100)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNetwork().to(device)

input_ = torch.randn(10,10)
logit = model(input_)
out = nn.Softmax(dim=1)(logit)
target = torch.empty(10,dtype=torch.long)

#print(input_.data.shape, target.data.shape)
print(input_)
print(target)
loss = nn.CrossEntropyLoss()#Crossentropy 
loss = torch.sqrt(loss(out,target))


