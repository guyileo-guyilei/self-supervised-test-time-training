import torch
from torch import nn

class ResNet(nn.Module):  # Control group for our experiment is a ResNet based on the SSResNet structure
    def __init__(self):
        super(ResNet, self).__init__()
        from models.SSResNet import SharedBranch, MainHead
        
        # Copy shared branch and main head from SSResNet class
        self.shared_branch = SharedBranch()
        self.main_head = MainHead()
        
    def forward(self,x):
        x = self.shared_branch(x)
        x = self.main_head(x)
        
        return x