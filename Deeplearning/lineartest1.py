import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class Minist(Dataset):
    def __init__(self,root,is_train = True):
        self.data_list = []
        sub_dir = "train" if is_train else "test"
        for tag in os.listdir(f'{root}/{sub_dir}'):
            img_dir = f'{root}/{sub_dir}/{tag}'
            for img_filename in os.listdir(img_dir):
                img_path = f'{img_dir}/{img_filename}'
                self.data_list.append((img_path,tag))
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self,index):
        self.data_imge = self.data_list[index]
        image = cv2.imread(self.data_imge[0])

        image = image/255
        image = np.transpose(image,[2,0,1])
        #image = np.expand_dims(image,0)  #增加维度0轴前


        image = torch.tensor(image,dtype=torch.float32)
        #image = torch.tensor(image.reshape(-1),dtype=torch.float32)
        tag_onehot = torch.zeros(2).scatter_(0,torch.tensor(int(self.data_imge[1])),1).reshape(-1)
        return image,tag_onehot

if __name__ == '__main__':
    minist = Minist(r"catdog_data")

    print(minist[8000][1])
