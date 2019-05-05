import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Function
from pytorch.slice import SliceFunction


class FusionFunction(Function):

    @staticmethod
    def forward(ctx, local, glbal):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class AssembleFunction(Function):

    @staticmethod
    def forward(ctx, full, coeff):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class GuidanceMap(nn.Module):

    def __init__(self, input, output):
        pass

    def forward(self, input):
        pass


class Network(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()
        self.S1 = nn.Conv2d(4, 8, 3, 2)
        self.S2 = nn.Conv2d(8, 16, 3, 2)
        self.S3 = nn.Conv2d(16, 32, 3, 2)
        self.S4 = nn.Conv2d(32, 64, 3, 2)
        self.L1 = nn.Conv2d(64, 64, 3)
        self.L2 = nn.Conv2d(64, 64, 3)
        self.G1 = nn.Conv2d(64, 64, 3, 2)
        self.G2 = nn.Conv2d(64, 64, 3, 2)
        self.G3 = nn.Linear(256, 256)
        self.G4 = nn.Linear(256, 128)
        self.G5 = nn.Linear(128, 64)

    def forward(self, full):
        low = nn.Upsample((256, 256))(full)

        # low-level feature
        low = func.relu(self.S1(low))
        low = func.relu(self.S2(low))
        low = func.relu(self.S3(low))
        low = func.relu(self.S4(low))

        # local path
        local = func.relu(self.L1(low))
        local = func.relu(self.L2(local))

        # global path
        glbal = func.relu(self.G1(low))
        glbal = func.relu(self.G2(glbal))
        glbal = func.relu(self.G3(glbal))
        glbal = func.relu(self.G4(glbal))
        glbal = func.relu(self.G5(glbal))

        # fuse both path
        coeff = FusionFunction.apply(local, glbal)

        # upsampleing
        guide = GuidanceMap(full)
        coeff = SliceFunction.apply(coeff, guide)

        # assemble
        output = AssembleFunction.apply(full, coeff)

        return output
