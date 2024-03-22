import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointconv_utils import PointConvDensitySetAbstraction, PointConvDensitySetPropagation


class get_model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(get_model, self).__init__()
        self.sa0 = PointConvDensitySetAbstraction(npoint=2048, nsample=32, in_channel=in_channels,
                                                  mlp=[16, 24, 32], bandwidth=0.05, group_all=False)
        self.sa1 = PointConvDensitySetAbstraction(npoint=1024, nsample=32, in_channel=32,
                                                  mlp=[32, 48, 64], bandwidth=0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=256, nsample=32, in_channel=64,
                                                  mlp=[64, 96, 128], bandwidth=0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=64, nsample=32, in_channel=128,
                                                  mlp=[128, 196, 256], bandwidth=0.4, group_all=False)
        self.sa4 = PointConvDensitySetAbstraction(npoint=16, nsample=32, in_channel=256,
                                                  mlp=[256, 384, 512], bandwidth=0.8, group_all=False)

        self.fp4 = PointConvDensitySetPropagation(nsample=16, in_channel=512, mlp=[512,256], bandwidth=0.8, group_all=False)
        self.fp3 = PointConvDensitySetPropagation(nsample=16, in_channel=256, mlp=[256,128], bandwidth=0.4, group_all=False)
        self.fp2 = PointConvDensitySetPropagation(nsample=16, in_channel=128, mlp=[128,64], bandwidth=0.2, group_all=False)
        self.fp1 = PointConvDensitySetPropagation(nsample=16, in_channel=64, mlp=[64,32], bandwidth=0.1, group_all=False)
        self.fp0 = PointConvDensitySetPropagation(nsample=16, in_channel=32, mlp=[32,32], bandwidth=0.05, group_all=False, last_stage=True)

        self.conv1 = nn.Conv1d(32, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(32, num_classes, 1)

    def forward(self, xyz):
        l_points = xyz
        l_xyz = xyz[:, :3, :]

        l0_xyz, l0_points = self.sa0(l_xyz, l_points)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        l_points = self.fp0(l_xyz, l0_xyz, None, l0_points)

        last_features = self.drop1(F.relu(self.bn1(self.conv1(l_points))))
        x = self.conv2(last_features)
        # x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, last_features


class get_ce_loss(nn.Module):
    def __init__(self):
        super(get_ce_loss, self).__init__()

    def forward(self, pred, target, trans_feat):

	ce_loss = F.cross_entropy(pred, target)
	return ce_loss


if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,4,8192))
    # label = torch.randn(8,16)
    model = get_model(4,8)
    output1, output2 = model(input)
    print(output1.size())
