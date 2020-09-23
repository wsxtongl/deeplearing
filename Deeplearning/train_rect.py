import torch
import os
import torch.nn as nn
from  torch import optim
from draw_rect import dataset
import net
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image,ImageDraw
from matplotlib import pyplot as plt
DEVICE = "cuda:0"

class train():
    def __init__(self,root):
        self.data_set = dataset(root)
        self.net = net.cnn_net()

        self.net.load_state_dict(torch.load("train_rectpoint/1.t"))

        self.net.to(DEVICE)
        self.opt = optim.Adam(self.net.parameters())
        self.loss_func = nn.MSELoss()
        self.train_dataloader = DataLoader(self.data_set, batch_size=10, shuffle=True)
        self.test_dataloader = DataLoader(self.data_set, batch_size=10, shuffle=False)
    def iou(self,rec1,rec2):
        s_rec1 = (rec1[2] -rec1[0])*(rec1[3]-rec1[1])
        s_rec2 = (rec2[2] - rec2[0])*(rec2[3]-rec2[1])
        s_sum = s_rec1 + s_rec2
        #左上点都取最大
        w1 = max(rec1[0],rec2[0])
        h1 = max(rec1[1], rec2[1])
        #右下角都取最小
        w2 = min(rec1[2],rec2[2])
        h2 = min(rec1[3],rec2[3])
        w = w2 - w1
        h = h2 - h1
        #相交矩形面积
        intersect = w * h
        if intersect <= 0:
            return 0
        else:
            iou = intersect/(s_sum-intersect)
            return iou
    def __call__(self,):

        Train = True
        while True:
            plt.ion()
            if Train:
                for epoch in range(0):
                    loss_sum = 0.
                    for i,(x_,y_) in enumerate(self.train_dataloader):
                        img = x_.permute(0,3,1,2)
                        img = img.to(DEVICE)
                        y_ = y_.to(DEVICE)
                        img = self.net(img)
                        loss = self.loss_func(y_,img)
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                        loss_sum += loss.detach().item()
                    aver_loss = loss_sum/len(self.train_dataloader)
                    print("epoch",epoch,"   ","aver_loss",aver_loss)
                #torch.save(self.net.state_dict(),"./train_rectpoint/{}.t".format(1))
                for i, (x, y) in enumerate(self.test_dataloader):
                    plt.cla()

                    x_ = x[0].numpy()

                    x = x.permute(0,3,1,2)
                    x = x.to(DEVICE)
                    out = self.net(x)
                    out = out.cpu().detach()
                    out = out[0].numpy() * 300
                    y = y[0].numpy()*300
                    out_iou = list(np.squeeze(out))
                    y_iou = list(np.squeeze(y))
                    iou = self.iou(y_iou,out_iou)
                    print(iou)
                    img_data = np.array((x_ + 0.5) * 255, dtype=np.int8)
                    img = Image.fromarray(img_data, "RGB")
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(y, outline="red", width=2)
                    draw.rectangle(out, outline="blue", width=2)

                    plt.imshow(img)
                    plt.xticks([])
                    plt.yticks([])
                    plt.pause(0.5)
                    plt.show()
            plt.ioff()
if __name__ == '__main__':
    path1 = r"D:\BaiduNetdiskDownload\20200723_深度学习05\test_20200723_02\data"
    train = train("./scenery")
    train()
