import random
import torch
from torch.backends import cudnn
from torchvision.transforms import ToPILImage
from utils import Realistic_Projection, accuracy, Transformations
from clip import clip
from torch import nn
import timm
from torch.cuda.amp import GradScaler, autocast
from loss import *


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
        super(Depth_Contrastive_Encoder, self).__init__()
        self.image_encoder = timm.create_model(args.depth_encoder_backbone_name, pretrained=False)
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


class Multi_Views_Contrastive_Encoder(nn.Module):
    def __init__(self):
        super(Multi_Views_Contrastive_Encoder, self).__init__()
        self.image_encoder = timm.create_model(args.multi_views_encoder_backbone_name, pretrained=False)
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


class DepthPointCLIP:
    def __init__(self, classname):
        # # Loading CLIP
        print("====================================================================================")
        print(f"Loading {args.clip_backbone_name} CLIP")
        self.clip = load_clip_to_cpu(clip_backbone_name=args.clip_backbone_name)
        self.clip.to(args.device)
        self.dtype = self.clip.dtype
        print(f"Loading {args.clip_backbone_name} CLIP Successfully")
        print("====================================================================================")

        #  Text Encoder from CLIP
        self.text_encoder = Text_Encoder(clip=self.clip, classname=classname)

        # Image Encoder from CLIP
        self.image_encoder = self.clip.visual

        # Realistic_Projection from PointCLIP V2
        self.projection = Realistic_Projection()


        self.dtype = torch.float16

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
        if is_continue is True:
            self.depth_contrastive_encoder = torch.load('depth.pt').to(args.device)
        cudnn.benchmark = True
        self.depth_contrastive_encoder.to(args.device)
        self.depth_contrastive_encoder.train()

        # scaler
        scaler = GradScaler()

        # optimzier
        # optimizer = torch.optim.Adam(self.depth_contrastive_encoder.parameters(), lr=args.lr_dep,
        #                              weight_decay=args.weight_decay_dep)
        optimizer = torch.optim.SGD(self.depth_contrastive_encoder.parameters(), lr=args.lr_dep)

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0,
                                                               last_epoch=-1)
        # Cross_Entropy
        criterion = torch.nn.CrossEntropyLoss()

        # transform
        transfrom = Transformations()

        # depth_contrastive_encoder
        print("====================================================================================")
        print('Training Depth Contrastive Encoder')

        for epoch in range(args.epochs_dep):
            for pc, y in dataloader:
                pc = pc.to(args.device)
                imgs = self.project(pc)  # shape(batch_size * views, channels, height, weight)
                imgs = F.unfold(imgs, kernel_size=args.patch_size, stride=args.patch_size)
                imgs = imgs.transpose(2, 1).contiguous()
                imgs = imgs.view(args.batch_size * args.views * (224 * 224) // (args.patch_size ** 2), 3,
                                 args.patch_size, args.patch_size)
                # shape(batch_size * views * HW/patch_size^2, c, patch_size, patch_size)

                # transform
                imgs1 = transfrom(imgs)

                # train
                with autocast():
                    z = self.depth_contrastive_encoder(imgs)
                    logits, labels = nt_xent_loss(z, z.shape[0]/2, 2, args.temperature_dep)
                    loss = criterion(logits, labels)
                    print(loss)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # save model
                torch.save(self.depth_contrastive_encoder, 'depth.pt')

            acc = accuracy(logits, labels, (1, 5))
            print(f'The {epoch+1}-th epochs, loss = {loss}, acc = {acc}')

            # warm-up
            if epoch >= 10:
                scheduler.step()

        print('Train Depth Contrastive Encoder successfully')
        print("====================================================================================")

    def train_multi_views_contrasitive_encoder(self, dataloader, is_continue=False):
        if is_continue is True:
            self.multi_views_contrasitive_encoder = torch.load('depth.pt').to(args.device)

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
