import torch
import torch.nn.init # for xavier init
from torch.autograd import Variable

import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import random
torch.manual_seed(777)  # reproducibility

