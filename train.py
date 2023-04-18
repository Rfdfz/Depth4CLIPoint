from ModelNet import ModelNetDataset
from model import *
from torch.utils.data.dataloader import DataLoader
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    dataset = ModelNetDataset(npoints=1024, scale=10)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)
    classnames = dataset.classnames

    # train depth contrastive encoder
    model = Depth4CLIPoint(classnames)
    model.train_depth_contrastive_encoder(dataloader, is_continue=True)

    # train muti views contrastive encoder


