import os
import torch
from torch.utils.data import Dataset ,DataLoader
import cv2
import numpy as np
from torch import optim
import torch.nn as nn

DEVICE = "cuda:0"

class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(25 * 25 * 64, 1024),
            torch.nn.ReLU(),
            #torch.nn.Dropout(p=0.2),
            torch.nn.Linear(1024, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 2),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 25 * 25 * 64)
        x = self.dense(x)
        return x

class minist(Dataset):
    def __init__(self,root,is_train = True):

        self.data_list = []
        sub_dir = "train" if is_train else "test"
        for tag in os.listdir(f'{root}/{sub_dir}'):
            img_dir = f'{root}/{sub_dir}/{tag}'
            for img_filename in os.listdir(img_dir):
                img_path = f'{img_dir}/{img_filename}'
                self.data_list.append((img_path, tag))
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        data = self.data_list[index]
        image = cv2.imread(data[0])
        image = image/255
        image = torch.tensor(np.transpose(image,[2,0,1]),dtype=torch.float32)
        #转one-hot
        label = torch.zeros(2).scatter_(0,torch.tensor(int(data[1])),1).reshape(-1)
        return image,label


class train():
    def __init__(self,root):
        self.train_dataset = minist(root)
        self.test_dataset = minist(root,is_train=False)
        self.train_dataloader = DataLoader(self.train_dataset,batch_size=100,shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset,batch_size=100,shuffle=False)
    def __call__(self):
        self.net = Net4()
        self.net.to(DEVICE)
        self.opt = optim.Adam(self.net.parameters())
        for epoch in range(10000):
            loss_sum = 0
            test_loss_sum = 0
            for i , (y_,label) in enumerate(self.train_dataloader):
                y_, label = y_.to(DEVICE), label.to(DEVICE)
                y_pred = self.net(y_)
                loss = torch.mean((y_pred - label)**2)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                loss_sum+=loss.cpu().detach().item()
                print(i)
            train_average_loss = loss_sum/len(self.train_dataloader)
            sum_score = 0

            for i, (y_, label) in enumerate(self.test_dataloader):
                label = label.to(DEVICE)
                y_ = y_.to(DEVICE)
                y_pred = self.net(y_)
                loss = torch.mean((y_pred - label) ** 2)
                test_loss_sum += loss.cpu().detach().item()
                y_index = torch.argmax(y_pred,dim=1)
                label_index = torch.argmax(label,dim=1)
                sum_score += torch.sum(torch.eq(y_index,label_index).float())
            test_average_loss = test_loss_sum/len(self.test_dataloader)
            test_score = sum_score/len(self.test_dataset)
            print("epoch",epoch,"train_loss",train_average_loss,"test_loss",test_average_loss
                  ,"test_score",test_score)
if __name__ == '__main__':
    train = train(r"catdog_data")
    train()
