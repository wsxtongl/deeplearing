import os
import torch
from torch.utils import data
from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
class dataset(data.Dataset):
    def __init__(self,path):
        self.path = path

        self.dataset = os.listdir(path)

        self.dataset.extend(os.listdir(path))

        # for filename in os.listdir(self.path):
        #      self.dataset.append(self.path + '/' + filename)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path,self.dataset[index]))

        img_data = torch.tensor(np.array(img) / 255 - 0.5,dtype=torch.float32)

        str = self.dataset[index].split(".")[1:5]

        label = torch.tensor(np.array(str,np.float32)/300)


        return img_data,label


if __name__ == '__main__':
    plt.ion()

    mydata = dataset("./scenery")
    dataloader = data.DataLoader(dataset=mydata,batch_size=1,shuffle=False)
    for i,(x,y) in enumerate(dataloader):
        plt.cla()
        print(i, x.shape, y.shape)
        x = x[0].numpy()
        y = y[0].numpy()

        img_data = np.array((x+0.5)*255,dtype=np.int8)

        img = Image.fromarray(img_data,"RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle(y*300,outline="red",width=2)
        #img.save("./picture/{0}.jpg".format(i))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.5)
        plt.show()
    plt.ioff()