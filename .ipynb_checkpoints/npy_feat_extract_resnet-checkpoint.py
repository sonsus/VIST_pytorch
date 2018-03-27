import torch 
from torch import nn
import torch.nn.init # for initialization
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
#import torchvision.models as models
import re_models


import matplotlib.pyplot as plt
import numpy as np

import dataset as d
import util as u
from pathlib import Path

import pickle
import h5py # making dataset (or db) file maybe    

class opt:
    def __init__(self):
        self.resnetname_num="resnet50"
        self.datapath=Path("./data/")
        self.batch_size=1000 
        self.keys=["train","val","test"]
    def print_settings(self):
        print(self.resnetname_num)
        print(self.datapath)
        print(self.batch_size)
        print(self.keys)
        return None
    
        
class MakeFeatureNPY(opt):
    def __init__(self,opt):
        self.resnetname_num=opt.resnetname_num
        self.datapath=opt.datapath
        self.batch_size= opt.batch_size
        self.keys=opt.keys
        self.ds_dict=self.get_ds_dict()
        self.loader_dict=self.make_loaders()
        self.model=self.get_model()
        
        
    def get_ds_dict(self):
        #datasets
        vist_train=d.ImgDset_Folder(self.datapath / "train")  
        vist_val=d.ImgDset_Folder(self.datapath / "val")  
        vist_test=d.ImgDset_Folder(self.datapath / "test")  

        ds_dict=dict()
        for key in self.keys:
            ds_dict[key]=eval("vist_{}".format(key))

        return ds_dict

    def make_loaders(self):
        #dataloaders
        train_loader=DataLoader(dataset=self.ds_dict["train"], batch_size=self.batch_size, shuffle=False, num_workers=1)
        val_loader=DataLoader(dataset=self.ds_dict["val"], batch_size=self.batch_size, shuffle=False, num_workers=1)
        test_loader=DataLoader(dataset=self.ds_dict["test"], batch_size=self.batch_size, shuffle=False, num_workers=1)

        loader_dict=dict()
        for key in self.keys:
            loader_dict[key]=eval("{}_loader".format(key))
        return loader_dict

    def make_id2idx_pkl(self): #id2idx are sharable regardless model type
        #id2idx
        #pkl file!
        vist_train=self.ds_dict["train"]
        vist_val=self.ds_dict["val"]
        vist_test=self.ds_dict["test"]

        train_id2idx=dict(zip(vist_train.img_names, range(len(vist_train))))
        val_id2idx=dict(zip(vist_val.img_names, range(len(vist_val))))
        test_id2idx=dict(zip(vist_test.img_names, range(len(vist_test))))

        with open("train_id2idx.pkl","wb") as train:
            pickle.dump(train_id2idx, train)
        with open("val_id2idx.pkl","wb") as val:
            pickle.dump(val_id2idx, val)
        with open("test_id2idx.pkl","wb") as test:
            pickle.dump(test_id2idx, test)


    #model: 
    def get_model(self): #str: resnet50, resnet101, resnet152 could be possible inputs
        model=eval("re_models.{}".format(self.resnetname_num))(pretrained=True) #pretrained on Imagenet
        model=model.cuda()
        return model 


    #create h5 file obj
    def create_npy(self):
    #def create_h5pyfile(resnetname_num, datasets_dict=datasets_dict, loader_dict=loader_dict): #resnetname_num--> str: resnet50, resnet101, resnet152 could be possible inputs
        

        vist_train=self.ds_dict["train"]
        vist_val=self.ds_dict["val"]
        vist_test=self.ds_dict["test"]
        
        #create empty npy for dataset saving
        npy_train=np.zeros(shape=(len(vist_train),2048), dtype=np.float32)
        npy_val=np.zeros(shape=(len(vist_val),2048), dtype=np.float32)
        npy_test=np.zeros(shape=(len(vist_test),2048), dtype=np.float32)


        for key in self.keys:
            loader = self.loader_dict[key]
            idx=0
            for imgbatch in loader:
                X=Variable(imgbatch, volatile=True).cuda()
                featurebatch=self.model(X)
                eval("npy_{}".format(key))[idx:idx+self.batch_size]=featurebatch.data.cpu().numpy()
                print(idx)
                idx+=self.batch_size
            print("{name} done!".format(name=key))
            np.save("{ds}_{model}.npy".format(ds=key, model=self.resnetname_num), eval("npy_{}".format(key)))
            print("{ds}_{model}.npy saved!".format(ds=key, model=self.resnetname_num))
        return None


if __name__=="__main__":
    options=opt()
    for resnet in [ "resnet101", "resnet50", "resnet152"]:
        options.print_settings()
        options.resnetname_num=resnet

        make_f_extractor=MakeFeatureNPY(options)
        make_f_extractor.create_npy()




