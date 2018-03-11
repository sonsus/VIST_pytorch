import numpy as np
import pathlib as p #.is_dir(), .exists(), 
from pathlib import Path
from os import listdir, getcwd
from util import * #is_imgfile(), load_img():224,224 rgb conversion done

import torch.utils.data as dat
import torchvision.transforms as tr


class ImgDset_Folder(dat.Dataset):
    def __init__(self, targetpath=Path("./data")):
        super(ImgDset_Folder,self).__init__()
        self.img_path = Path(targetpath)
        self.img_names= [name for name in listdir(self.img_path) if is_imgfile(name)]
        
        normalize = tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        
        transform_list = [tr.ToTensor(), normalize]
        self.transform=tr.Compose(transform_list)

    def __getitem__(self, index):#used for dataset[i]
        img = load_img( self.img_path / self.img_names[index] )
        imgtensor = self.transform(img)
        return imgtensor

    def __len__(self):#used for len(dataset)
        return len(self.img_names)
    '''
    useful attributes:
        .img_path : where this img is loaded from
        .img_names: img file names loaded 
    '''