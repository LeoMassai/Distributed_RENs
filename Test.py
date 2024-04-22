from Models import REN

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import nn

n = 3
m = 2
p = 4
l = 6

RENsys = REN(m, p, n, l, bias=False, mode="l2stable")

a, b = RENsys(torch.tensor([1.6, 4.4], device=device), torch.tensor([4.8, 7.3, 4.2], device=device), 0, 0.7)

