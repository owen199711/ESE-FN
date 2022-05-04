import torch.nn as nn
import torch
import numpy as np;
class C_Net(nn.Module):
    def __init__(self,channel=256,reduction=16,kernel=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv2d(in_channels=channel, out_channels=2*channel, kernel_size=(1, 1), padding=0, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),#
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction , channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y =self.conv(x).view(b,c,-1,1)
        y=self.avg_pool(y).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class M_Net(nn.Module):
    def __init__(self,in_channel, channel, reduction=16):
        super(M_Net, self).__init__()
        self.conv_1=nn.Sequential(
            nn.Conv2d(in_channel, 4, kernel_size=(3, 1), stride=2),
            nn.Conv2d(4, 8, kernel_size=(7, 1), stride=2),
            nn.Conv2d(8, 22, kernel_size=(15, 1), stride=2)
        )
        self.avg_1=nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*11, channel*11//reduction, bias=False),#,bias=False
            nn.ReLU(inplace=True),
            nn.Linear(channel*11 // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y=self.conv_1(x)
        y=self.avg_1(y).view(-1,c*11)
        y=self.fc(y)
        b, out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        y= x*y.expand_as(x)
        return y


class Fusion_Net(nn.Module):
    def __init__(self,in_size_a,in_size_b,num_classes):
        super().__init__()
        self.a_size = in_size_a
        self.b_size = in_size_b
        self.con_size = min(in_size_a, in_size_b)
        self.layer_1=M_Net(2,2,reduction=8)
        self.layer_2=C_Net(channel=self.con_size,reduction=16,kernel=2)
        self.fc=nn.Linear(self.con_size,num_classes)
        self.down=nn.Linear(in_size_a,self.con_size)
    def forward(self,x):
        x_1 = x[:, 0:self.a_size]  # b 2048
        x_2 = x[:, self.a_size:]  # b 256
        if self.a_size>self.b_size:
            x_1=self.down(x_1)
        x_r=x_1.view(-1,self.con_size,1)
        x_s=x_2.view(-1,self.con_size,1)
        x_input=torch.cat((x_r,x_s),dim=-1).unsqueeze(3).permute(0,2,1,3) #8 2 256 1
        x_out_1=self.layer_1(x_input).permute(0,2,1,3) #8 256 2 1
        x_out_1=self.layer_2(x_out_1).squeeze(-1)
        x_out=torch.sum(x_out_1,dim=-1)
        out_socre=self.fc(x_out) #8 55
        return out_socre

if __name__ == '__main__':
    input = torch.randn(8, 256, 2, 1)
    c=C_Net(256,16,2);
    c(input);
