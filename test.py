import numpy as np
from libKMCUDA import kmeans_cuda
import torch
import torch.nn.functional as F
from torch.nn import MSELoss

arr = np.array([[0.5, 0.1, 0], [0.5, 0.4, 0]])
target = torch.tensor(np.zeros(arr.shape).astype(np.float))
print(arr.shape, target.shape)
print(arr, target)
print(F.cross_entropy(input=torch.tensor(arr), target=target))
mseloss = torch.nn.MSELoss(reduction='sum')
print(mseloss(torch.tensor(arr), target))