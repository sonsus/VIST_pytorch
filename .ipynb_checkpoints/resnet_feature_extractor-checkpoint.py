import torch 
from torch import nn
import torch.nn.init # for initialization
from torch.autograd import Variable


import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
import random
import dataset as d
import util as u

# transformation for pretrained models 



#ImageFolder dataset
#all jpgs need to be wrapped by the directory of having the same number
vist_train=d.(root="./data/images/train", 
                             transform= trans4pretrained)
vist_val=dsets.ImageFolder(root="./data/images/val")
vist_test=dsets.ImageFolder(root="./data/test",
                            transform= trans4pretrained)

#batch_size=1 

data_loader = torch.utils.data.DataLoader(dataset=vist_train,
                                          batch_size=1,                 #no batch needed
                                          shuffle=False,                #no need to shuffle: feature extraction
                                          num_workers=1)

#model here
Resnet=models.resnet50(pretrained=True) #pretrained on "ImageNet", 
                                        #expects identically normalized/shaped inputs

Resnet.cuda()
'''
model=Sequential(models.resnet50(pretrained = True ),
                 )

costf=nn.CrossEntropyLoss()
lr=0.001
train_epoch = 5
optimizer = torch.optim.Adam(Resnet.parameters(), lr=lr)

'''

#for epoch in range(train_epoch): 
    #avg_cost=0
    #tot_batch=int(len(vist_train)/batch_size)
len_dataset=\len(data_loader)

for i, images in enumerate(data_loader): # if it was training, I need Y as an label
    fnames_list=\implement! jpg wrapper, os walk? do i need jpg wrapper?
    X=Variable(images).cuda() # do I need Var here? I dont back prop
    feature=Resnet(X)
    print("image {fname} feature extracted as res50_{fname}.hdf5".format(fname=fnames_list[i]))

    \pickle? save?




