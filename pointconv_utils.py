"""
Utility function for PointConv
Originally from : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/utils.py
Modify by Wenxuan Wu
Date: September 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from sklearn.neighbors.kde import KernelDensity


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, nsample, xyz, points, density_scale=None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    # print(density_scale.shape, idx.shape)
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density


def grouping(feature, K, src_xyz, q_xyz, use_xyz=True):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    '''
    B, N, C = src_xyz.shape
    npoint = q_xyz.shape[1]
    point_indices = knn_point(K, src_xyz, q_xyz)
    grouped_xyz = index_points(src_xyz, point_indices)  # [B, npoint, nsample, C]
    # print(grouped_xyz.shape, q_xyz.shape)
    grouped_xyz_norm = grouped_xyz - q_xyz.view(B, npoint, 1, C)
    grouped_features = index_points(feature, point_indices)

    if use_xyz:
        new_points = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
    else:
        new_points = grouped_features

    return grouped_xyz_norm, new_points, point_indices



    # batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
    # idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    # idx.set_shape([batch_size, npoint, K, 2])
    #
    # grouped_xyz = tf.gather_nd(src_xyz, idx)
    # grouped_xyz -= tf.tile(tf.expand_dims(q_xyz, 2), [1, 1, K, 1])  # translation normalization
    #
    # grouped_feature = tf.gather_nd(feature, idx)
    # if use_xyz:
    #     new_points = tf.concat([grouped_xyz, grouped_feature], axis=-1)
    # else:
    #     new_points = grouped_feature
    #
    # return grouped_xyz, new_points, idx


def sample_and_group_all(xyz, points, density_scale=None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # new_xyz = torch.zeros(B, 1, C).to(device)
    new_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.view(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density


def group(nsample, xyz, points):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    # import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim=-1)

    return xyz_density


class DensityNet(nn.Module):
    def __init__(self, hidden_unit=[16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale = bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)

        return density_scale


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))

        return weights


class PointConvSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, group_all):
        super(PointConvSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, grouped_xyz_norm, _ = sample_and_group(self.npoint, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                self.npoint,
                                                                                                                -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointConvDensitySetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / xyz_density

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(xyz, points,
                                                                                          inverse_density.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(self.npoint, self.nsample, xyz,
                                                                                         points,
                                                                                         inverse_density.view(B, N, 1))
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # print(new_points.shape)
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        inverse_max_density = grouped_density.max(dim=2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        new_points = new_points * density_scale

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # print(new_points.shape,grouped_xyz.shape, weights.shape)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                self.npoint,
                                                                                                                -1)
        # print(new_points.shape)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointConvDensitySetPropagation(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bandwidth, group_all, last_stage=False):
        super(PointConvDensitySetPropagation, self).__init__()
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel+3
        for i,out_channel in enumerate(mlp):
            if i==0:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, (nsample,1)))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel if last_stage else out_channel+out_channel//2
            elif i==1:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.bandwidth = bandwidth
        self.group_all = group_all

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        grouped_xyz_norm, new_points, point_indices = grouping(interpolated_points, self.nsample, xyz1, xyz1)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # print(new_points.shape)
        # new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        # for i, conv in enumerate(self.mlp_convs):
        #     bn = self.mlp_bns[i]
        #     new_points = F.relu(bn(conv(new_points)))
        # print(new_points.shape)
        xyz_density = compute_density(xyz1, self.bandwidth)
        inverse_density = (1.0 / xyz_density).view(B, N, 1)
        # print(inverse_density.shape, point_indices.shape)
        grouped_density = index_points(inverse_density, point_indices)
        inverse_max_density = grouped_density.max(dim=2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        # print(new_points.shape, density_scale.shape)
        new_points = new_points * density_scale

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # print(new_points.shape,grouped_xyz.shape, weights.shape, N)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1))
        # print(new_points.shape)

        new_points = new_points.permute(0, 2, 3, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            if i==0:
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))

        # print(new_points.shape, points1.shape)
        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([new_points, torch.unsqueeze(points1, dim=2)], dim=1)
        else:
            new_points = new_points
        # print(new_points.shape)
        # new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            if i==1:
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
        new_points=torch.squeeze(new_points, dim=2)
        return new_points
