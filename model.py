from torch.backends import cudnn
from torchvision import models
from utils import Realistic_Projection, Transformations
from clip import clip
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from loss import *
import time
import os


def load_clip_to_cpu(clip_backbone_name):
    url = clip._MODELS[clip_backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = clip.build_model(state_dict or model.state_dict())
    return model


class Text_Encoder(nn.Module):
    def __init__(self, clip, classname):
        super(Text_Encoder, self).__init__()
        self.clip = clip
        self.dtype = clip.dtype
        self.classname = classname

    def forward(self):
        prompts = args.TEMPLATES.format(cls.replace('_', '') for cls in self.classname)
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.cuda()
        ft = self.clip.encode_text(prompts).repeat(1, args.views)
        return ft


class Depth_Contrastive_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = models.get_model(name=args.depth_encoder_backbone_name, num_classes=512,
                                        zero_init_residual=True)

        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(512, affine=False))  # output layer

        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN
        self.predictor = nn.Sequential(nn.Linear(512, 512, bias=False),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(512, 512))  # output layer

    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


class Multi_Views_Contrastive_Encoder(nn.Module):
    def __init__(self):
        super(Multi_Views_Contrastive_Encoder, self).__init__()
        self.encoder = models.__dict__[args.multi_views_encoder_backbone_name](2048, zero_init_residual=True)
        self.projector = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=512, bias=True)
        )

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.projector(x)
        return x


class Filiter(nn.Module):
    def __init__(self):
        super(Filiter, self).__init__()


class Depth4CLIPoint:
    def __init__(self, mode, classname):
        if mode not in ['train', 'test']:
            raise RuntimeError("Please input 'train' or 'eval'")
        if mode == 'test':
            # # Loading CLIP
            print("====================================================================================")
            print(f"Loading {args.clip_backbone_name} CLIP")
            self.clip = load_clip_to_cpu(clip_backbone_name=args.clip_backbone_name)
            self.clip.to(args.device)
            self.dtype = self.clip.dtype
            print(f"Loading {args.clip_backbone_name} CLIP Successfully")
            print("====================================================================================")
        else:
            self.dtype = torch.float16

        #  Text Encoder from CLIP
        self.text_encoder = Text_Encoder(clip=self.clip, classname=classname)

        # Image Encoder from CLIP
        self.image_encoder = self.clip.visual

        # Realistic_Projection from PointCLIP V2
        self.projection = Realistic_Projection()

        # Depth_Contrastive_Encoder
        self.depth_contrastive_encoder = Depth_Contrastive_Encoder()
        self.depth_contrastive_encoder.dtype = self.dtype

        # self.transforms = [GaussianBlur(kernel_size=(3, 3)), RandomRotation(30), ]

        # Muti_Views_Contrasitive_Encoder
        self.multi_views_contrasitive_encoder = Multi_Views_Contrastive_Encoder()
        self.multi_views_contrasitive_encoder.dtype = self.dtype

        # Filiter
        self.filiter = None

    def project(self, pc, img_size=224):
        img = self.projection.get_img(pc).to(args.device)
        img = torch.nn.functional.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=True)
        return img

    def train_depth_contrastive_encoder(self, dataloader, is_continue=False):
        # scaler
        scaler = GradScaler()

        # optimzier
        optimizer = torch.optim.SGD(self.depth_contrastive_encoder.parameters(), lr=args.lr_dep,
                                    weight_decay=args.weight_decay_dep, momentum=args.momentum)

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0,
                                                               last_epoch=-1)
        # Cross_Entropy
        criterion = torch.nn.CosineSimilarity(dim=1).to(args.device)

        # transform
        transfrom = Transformations()

        if is_continue is True and os.path.exists(r'checkpoints/checkpoint_dep.pth'):
            checkpoint = torch.load(r'checkpoints/checkpoint_dep.pth')
            self.depth_contrastive_encoder.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epochs = checkpoint['epoch']
        else:
            start_epochs = 0

        cudnn.benchmark = True
        self.depth_contrastive_encoder.to(args.device)
        self.depth_contrastive_encoder.train()

        # depth_contrastive_encoder
        print("====================================================================================")
        print('Training Depth Contrastive Encoder')
        for epoch in range(start_epochs, args.epochs_dep):
            start_time = time.time()
            for pc, _ in dataloader:
                pc = pc.to(args.device)
                imgs = self.project(pc)  # shape(batch_size * views, channels, height, weight)
                imgs = F.unfold(imgs, kernel_size=args.patch_size, stride=args.patch_size)
                imgs = imgs.transpose(2, 1).contiguous()
                imgs = imgs.view(args.batch_size * args.views * (224 * 224) // (args.patch_size ** 2), 3,
                                 args.patch_size, args.patch_size)
                # shape(batch_size * views * HW/patch_size^2, c, patch_size, patch_size)

                # transform
                imgs1, imgs2 = transfrom(imgs), transfrom(imgs)
                del imgs

                # train
                with autocast():
                    p1, p2, z1, z2 = self.depth_contrastive_encoder(imgs1, imgs2)
                    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            end_time = time.time()
            # if epoch % 10 == 0:
            print(f'The {epoch + 1}-th epochs: loss = {loss}, time = {end_time-start_time}')
            # warm-up
            if epoch >= 10:
                scheduler.step()
            # save model
            checkpoint = {
                'model': self.depth_contrastive_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(checkpoint, r"checkpoints/checkpoint_dep.pth")

        print('Train Depth Contrastive Encoder successfully')
        print("====================================================================================")

    def train_multi_views_contrasitive_encoder(self, dataloader, is_continue=False):

        if is_continue is True:
            self.multi_views_contrasitive_encoder = torch.load('mv.pt').to(args.device)

        self.multi_views_contrasitive_encoder.to(args.device)
        self.multi_views_contrasitive_encoder.train()
        # scaler
        scaler = GradScaler()
        # optimzier
        optimizer = torch.optim.Adam(self.multi_views_contrasitive_encoder.parameters(), lr=args.lr_mv,
                                     weight_decay=args.weight_decay_mv)
        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0,
                                                               last_epoch=-1)
        # Cross_Entropy
        criterion = torch.nn.CrossEntropyLoss()

        # Multi-Views Contrasive Encoder
        print("====================================================================================")
        print('Training Multi-Views Contrasive Encoder')
        for epoch in range(args.epochs_mv):
            for pc, y in dataloader:
                pc = pc.to(args.device)
                imgs = self.project(pc)  # shape(batch_size * views, channels, height, weight)

                # train
                with autocast():
                    z = self.multi_views_contrasitive_encoder(imgs)
                    logits, labels = nt_xent_loss(z, args.batch_size, 10, args.temperature_mv)
                    loss = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # save model
                torch.save(self.multi_views_contrasitive_encoder, 'mv.pt')

            # print(f'The {epoch+1}-th epochs, loss = {loss}')

            # warm-up
            if epoch >= 10:
                scheduler.step()

        print('Train Multi-Views Contrasive Encoder successfully')
        print("====================================================================================")
