# device
device = 'cuda'

# datasets configs
# datasets name: [ModelNet40, ModelNet10]
datasets_name = 'ModelNet40'
num_workers = 6

# clip backbone name: ['RN50', 'RN101', 'ViT-B/16', 'ViT-B/32']
clip_backbone_name = 'ViT-B/16'

# prompts
TEMPLATES = "A depth map of {} point cloud."

# args of depth_contrastive_encoder
depth_encoder_backbone_name = 'resnet50'        # ['resnet50', 'resnet101']
epochs_dep = 100
temperature_dep = 0.07
lr_dep = 0.01
weight_decay_dep = 0.0001
momentum = 0.9
# patch_size 16, 32
patch_size = 32

# args of multi_views_contrasitive_encoder
multi_views_encoder_backbone_name = 'resnet50'        # ['resnet50', 'resnet101']
epochs_mv = 200
temperature_mv = 0.07
lr_mv = 0.03
weight_decay_mv = 0

# batch_size
batch_size = 4

# views of projection
views = 10


