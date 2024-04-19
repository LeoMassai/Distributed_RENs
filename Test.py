from Models import REN
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from os.path import dirname, join as pjoin
import torch
from torch import nn

n = 3
m = 2
p = 4
l = 6

RENsys = REN(m, p, n, l, bias=False, mode="l2stable")

a, b = RENsys(torch.tensor([1.6, 4.4]), torch.tensor([4.8, 7.3, 4.2]), 0)
