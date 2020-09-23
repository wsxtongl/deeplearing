import torch
from torch import nn

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(784,10))
    def forward(self,x):
        h = x @ self.w
        h = torch.exp(h)
        z = torch.sum(h,dim=1,keepdim=True)
        return h/z

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100,52)
        self.fc3 = nn.Linear(52,10)
        self.soft_max = nn.Softmax(dim=1)
    def forward(self,x):
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.fc3(h)
        out = self.soft_max(h)
        return out
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,52),
            nn.ReLU(),
            nn.Linear(52,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        return self.layer(x)
class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
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



class Net5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),

            )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(1024, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)

        x = x.view(-1, 14 * 14 * 128)

        x = self.dense(x)
        return x

class Net6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),

            )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(25 * 25 * 64, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(1024, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 4),
        )

    def forward(self, x):
        x = self.conv1(x)

        x = x.reshape(-1, 25 * 25 * 64)

        x = self.dense(x)
        return x
class cnn_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=2),

            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16,32,3,1,2),

            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 2),

            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 2),

            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.mlp_layers = nn.Sequential(
            nn.Linear(128*20*20,4)
        )

    def forward(self,x):
        cnn_out = self.cnn_layers(x)

        cnn_out = cnn_out.reshape(-1, 128 * 20 * 20)
        output = self.mlp_layers(cnn_out)
        return output
if __name__ == '__main__':
    net = cnn_net()
    x = torch.randn(1,3,300,300)
    print(net(x))