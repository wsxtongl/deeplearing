import torch
from lineartest1 import *
from  net import *
from torch.utils.data import DataLoader
from  torch import optim as opt
from torch.utils.tensorboard import SummaryWriter
import os



DEVICE = "cuda:0"

#启用tensorboard:tensorboard --logdir=logs
#http://localhost:6006/

class Train():
    #训练准备（获取数据）
    def __init__(self,root):

        #self.summartWriter = SummaryWriter("./logs")

        #训练集
        self.train_dataset = Minist(root)
        self.train_datalaoder = DataLoader(self.train_dataset,batch_size=100,shuffle=True)

        # 验证集
        self.test_dataset = Minist(root,is_train=False)
        self.test_datalaoder = DataLoader(self.test_dataset, batch_size=100, shuffle=False)

        #创建模型
        self.net = Net4()
        #加载训练参数
        #self.net.load_state_dict(torch.load("checkpoint/2.t"))
        self.net.to(DEVICE)
        self.opt = opt.Adam(self.net.parameters())

    #训练代码
    def __call__(self):
        for epoch in range(1):
            loss_sum = 0.
            for i,(img,tage) in enumerate(self.train_datalaoder):
                img,tage = img.to(DEVICE),tage.to(DEVICE)
                y = self.net(img)
                loss = torch.mean((tage-y)**2)

                #训练三件套
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                loss_sum+=loss.cpu().detach().item()
                print(i)
            avg_loss = loss_sum/len(self.train_datalaoder)

            #验证
            sum_score = 0.
            test_loss_sum = 0.
            for i,(img,tage) in enumerate(self.test_datalaoder):
                img,tage = img.to(DEVICE),tage.to(DEVICE)
                test_y = self.net(img)

                loss = torch.mean((tage-test_y)**2)
                test_loss_sum+=loss.cpu().item()
                #将one-hot转为标签值
                pre_tage = torch.argmax(test_y,dim=1)
                label_tage = torch.argmax(tage,dim=1)
                print(pre_tage)
                print("****")
                print(label_tage)
                sum_score +=torch.sum(torch.eq(pre_tage, label_tage).float())

            score = (sum_score/len(self.test_dataset))*100
            test_avg_loss =  test_loss_sum/ len(self.test_dataset)
            #self.summartWriter.add_scalars("loss",{"train_loss":avg_loss,"test_loss":test_avg_loss},epoch)
            #self.summartWriter.add_scalar("score",score,epoch)

            print(epoch,"train_loss:",avg_loss,"test_loss:",test_avg_loss,"score:",score.item())
            #torch.save(self.net.state_dict(),f"./checkpoint/{epoch}.t")
if __name__ == '__main__':
    train = Train(r"catdog_data")
    train()