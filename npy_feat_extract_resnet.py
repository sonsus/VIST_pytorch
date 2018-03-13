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
batch_size=1000
train_loader=DataLoader(dataset=vist_train, batch_size=batch_size, suffle=False, num_workers=1)
val_loader=DataLoader(dataset=vist_val, batch_size=batch_size, suffle=False, num_workers=1)
test_loader=DataLoader(dataset=vist_test, batch_size=batch_size, suffle=False, num_workers=1)

loader_list=[train_loader, val_loader, test_loader]

#id2idx
#pkl file!

train_id2idx=dict(zip(train_name_list, range(len(vist_test))))
val_id2idx=dict(zip(val_name_list, range(len(vist_val))))
test_id2idx=dict(zip(test_name_list, range(len(vist_test))))

with open("train_id2idx.pkl","wb") as train:
    train.dump(train_id2idx, train)
with open("val_id2idx.pkl","wb") as val:
    val.dump(val_id2idx, val)
with open("test_id2idx.pkl","wb") as test:
    test.dump(test_id2idx, test)

print(pickle.load(open("train_id2idx.pkl", "rb")))
print(pickle.load(open("val_id2idx.pkl", "rb")))
print(pickle.load(open("test_id2idx.pkl","rb")))

#model: resnet50
Resnet=models.resnet50(pretrained=True) #pretrained on Imagenet
Resnet=Resnet.cuda()

for i, loader in enumerate(loader_list):
    for imgbatch in loader:
        X=Variable(imgbatch, volatile=True).cuda()
        featurebatch=Resnet(X)
        namelist=['train', 'val', 'test']
        
        print(featurebatch.data.cpu().numpy().shape)
        np.save(namelist[i]+"resnet50.npy", featurebatch.data.cpu().numpy())
        
        print("{name} done!".format(name=namelist[i]))

