import torch
import torch.nn as nn

m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = torch.randn(3, 2, requires_grad=True)
print(input)
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 1])
print(target.data.shape)
output = loss(m(input), target)
output.backward()
