import torch
from loss import *
import args
from ModelNet import ModelNetDataset
from model import *
import tqdm
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = ModelNetDataset(npoints=2048, scale=10)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    classnames = dataset.classnames

    # train depth contrastive encoder
    model = DepthPointCLIP(classnames)
    model.train_depth_contrastive_encoder(dataloader, is_continue=False)

    # train muti views contrastive encoder


