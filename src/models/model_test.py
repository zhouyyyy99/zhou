import torch.nn as nn
import torch
import math

feat_dim = 116
d_model = 128


class net(nn.Module):
    
    def __init__(self, feat_dim, d_model):
        super(net, self).__init__()
        self.project_inp = nn.Linear(feat_dim, d_model)
    
    def forward(self, x):
        inp = x.permute(1, 0, 2)
        inp = self.project_inp(inp)
        print(inp.shape)
        inp = inp * math.sqrt(d_model)
        print(inp.shape)
        return inp
            

x = torch.randn((32, 140, 116))

net = net(feat_dim, d_model)

print(net(x).shape)