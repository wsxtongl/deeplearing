import torch
from torch import jit
from net import Net3

if __name__ == '__main__':
    model = Net3()
    model.load_state_dict(torch.load("checkpoint/2.t"))
    input = torch.randn(1,784)
    torch_model = jit.trace(model,input)
    torch_model.save("net.pt")