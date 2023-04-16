import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import numpy as np
from model import *
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def point_cloud_normalize(point_cloud):
    """
    :param point_cloud: point cloud data
    :return: normalized point cloud data
    """
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    m = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
    return point_cloud / m


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataset(Dataset):
    def __init__(self, root=None, npoints=1024, scale=10, split='train'):
        if root is None:
            self.root = 'modelnet40_normal_resampled'
        else:
            self.root = root
        if split != 'train' and split != 'test':
            raise Exception("Please input split with 'train' or 'test'")
        else:
            self.split = split
        if scale != 10 and scale != 40:
            raise Exception("Please input scale with 10 or 40")
        else:
            self.scale = scale
        self.npoints = npoints
        print("====================================================================================")
        print(f'Loading ModelNet{scale}')
        with open(os.path.join(self.root, 'modelnet' + str(self.scale) + '_shape_names.txt'), "r", encoding='utf-8') as f:
            self.classnames = [i[:-1] for i in f.readlines()]
        self.filelist = np.loadtxt(os.path.join(self.root, "filelist.txt"), dtype="str")
        self.example_name_list = np.loadtxt(
            os.path.join(self.root, "modelnet" + str(self.scale) + "_" + str(self.split) + ".txt"), dtype="str")

        # self.data_path_list = [(os.path.join(self.root, file), (file.split('/')[-1]).split('.')[0]) for file in
        #                        self.filelist if (file.split('/')[-1]).split('.')[0] in self.example_name_list]
        self.data_path_list = []
        for file in tqdm.tqdm(self.filelist):
            if (file.split('/')[-1]).split('.')[0] in self.example_name_list:
                self.data_path_list.append((os.path.join(self.root, file), (file.split('/')[-1]).split('.')[0]))
        print(f'Loading ModelNet{scale} Successfully')
        print("====================================================================================")

    def __len__(self):
        return len(self.example_name_list)

    def __getitem__(self, idx):
        point_cloud = np.loadtxt(self.data_path_list[idx][0], delimiter=',').astype(np.float16)
        label = self.data_path_list[idx][1]
        # point_cloud = farthest_point_sample(point_cloud, self.npoints)
        point_cloud[:, 0:3] = point_cloud_normalize(point_cloud[:, 0:3])
        point_cloud = torch.Tensor(point_cloud[:, 0:3])
        return point_cloud, label
