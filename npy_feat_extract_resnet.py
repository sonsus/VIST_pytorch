import torch 
from torch import nn
import torch.nn.init # for initialization
from torch.autograd import Variable
from torch.utils.data import DataLoader


import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np

import dataset as d
import util as u
from pathlib import Path

import pickle
#import h5py # making dataset (or db) file maybe    

#datasets
vist_train=d.ImgDset_Folder(Path("./data/train"))  
vist_val=d.ImgDset_Folder(Path("./data/val"))  
vist_test=d.ImgDset_Folder(Path("./data/test"))  

#dataloaders
train_loader=DataLoader(dataset=vist_train, batch_size=len(vist_train), suffle=False, num_workers=1)
val_loader=DataLoader(dataset=vist_val, batch_size=len(vist_val), suffle=False, num_workers=1)
test_loader=DataLoader(dataset=vist_test, batch_size=len(vist_test), suffle=False, num_workers=1)

loader_list=[train_loader, val_loader, test_loader]

#id2idx
train_id2idx=dict(zip(train_name_list, range(len(vist_test))))
val_id2idx=dict(zip(val_name_list, range(len(vist_val))))
test_id2idx=dict(zip(test_name_list, range(len(vist_test))))

#pkl file! 


#model: resnet50
Resnet=models.resnet50(pretrained=True) #pretrained on Imagenet
Resnet=Resnet.cuda()

for i, loader in enumerate(loader_list):
    for imgbatch in loader:
        X=Variable(imgbatch, volatile=True).cuda()
        featurebatch=Resnet(X)
        namelist=['train', 'val', 'test']
        np.save(namelist[i]+"resnet50.npy", featurebatch.data.cpu().numpy())
        # is this right?
