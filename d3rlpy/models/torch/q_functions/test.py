import torch
t = torch.tensor((1,2,3),dtype=float)
print(t)
y = torch.tensor((4,2,3),dtype=float)
print(y)
a = torch.mean(t-y)
print(a)
