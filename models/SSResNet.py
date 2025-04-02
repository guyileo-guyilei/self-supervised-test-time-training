import torch
from torch import nn

# ResNet individual layer, using [norm-relu-conv] architecture
class ResLayer(nn.Module):
    def __init__(self, input_channels, output_channels, downsample=False):
        super(ResLayer, self).__init__()
        self.downsample = downsample
        
        self.norm1 = nn.GroupNorm(4,input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels,output_channels,(3,3),padding=1)
        
        self.norm2 = nn.GroupNorm(8,output_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels,output_channels,(3,3),padding=1)
        
        self.pool = nn.MaxPool2d((2,2),stride=2)
        
        if downsample:
            stride = 2
        else:
            stride = 1
            
        self.Ws = nn.Conv2d(input_channels,output_channels,(1,1),stride=stride)

    def forward(self, x):
        res = x 
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.downsample:
            x = self.pool(x)
        return x + self.Ws(res)

# Combined Model (SSHead and MainHead)
class SharedBranch(nn.Module):  # Starting branch that encompasses all layers before forking
    def __init__(self):
        super(SharedBranch, self).__init__()        
        
        self.layer1 = ResLayer(4,16,downsample=True)
        self.layer2 = ResLayer(16,32,downsample=True)
        
        self.conv0 = nn.Conv2d(1,4,kernel_size=3,padding=1)
    
    def forward(self,x):
        x = self.conv0(x)
   
        x = self.layer1(x)
    
        x = self.layer2(x)
        
        return x
    
class MainHead(nn.Module):  # Testing head to stay constant at test time
    def __init__(self):
        super(MainHead, self).__init__()
        
        self.layer1 = ResLayer(32,64)
        self.layer2 = ResLayer(64,64)
        
        self.fc = nn.Linear(64,10)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        main = self.fc(torch.flatten(x, 1))
        
        return main

class SSHead(nn.Module):
    def __init__(self):
        super(SSHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.relu = nn.ReLU(inplace=False)
        
        self.fc = nn.Linear(64,4)  # SS is 4 way classification of angle
        
        self.layer1 = ResLayer(32,64)
        self.layer2 = ResLayer(64,64)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        ss = self.fc(torch.flatten(x, 1))
        
        return ss

class SSResNet(nn.Module):
    def __init__(self):
        super(SSResNet, self).__init__()
        
        self.shared_branch = SharedBranch()  # Using class to call parameters in optimizer later
        self.main_head = MainHead()
        self.ss_head = SSHead()

    def forward(self,x):  # Combine all branches and return two outputs, one for main head and one for ss train head
        x = self.shared_branch(x)

        main = self.main_head(x)
        
        ss = self.ss_head(x)

        return main, ss