import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from copy import deepcopy
from pi_seg.model.is_deepPI_model import DeepPI
from pi_seg.model import initializer
from config import device, PRE_TRAINED, PRE_TRAINED_PATH

def load_unet():
    # Load UNet with ResNet50 backbone
    unet = smp.Unet('resnet50', in_channels=1, encoder_weights=None, classes=1)
    
    """Load and customize the pre-trained UNet model."""
    if PRE_TRAINED:
        print(f"PRE_TRAINED is True. Loading weights from: {PRE_TRAINED_PATH}")

        state = torch.load(PRE_TRAINED_PATH, map_location='cpu')
        state_dict = state['state_dict']

        # Modify state dictionary keys
        unet_state_dict = deepcopy(state_dict)
        for k in list(unet_state_dict.keys()):
            unet_state_dict['encoder.' + k] = unet_state_dict[k]
            del unet_state_dict[k]

        unet.load_state_dict(unet_state_dict, strict=False)
        
        print("Pre-trained UNet weights loaded successfully.")

    ## Customize model layers
    encoder_conv1 = nn.Conv2d(64+1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    
    decoder3_conv1_0 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    decoder3_conv1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    decoder3_conv2_0 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    decoder3_conv2_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    unet.encoder.conv1 = encoder_conv1
    
    unet.decoder.blocks[3].conv1[0] = decoder3_conv1_0
    unet.decoder.blocks[3].conv1[1] = decoder3_conv1_1

    unet.decoder.blocks[3].conv2[0] = decoder3_conv2_0
    unet.decoder.blocks[3].conv2[1] = decoder3_conv2_1

    unet.decoder.blocks = unet.decoder.blocks[:4]

    # remove segmentation head
    unet.segmentation_head = nn.Sequential(*[])

    return unet

def initialize_model():
    """Initialize the DeepPI model with a modified UNet backbone."""
    unet = load_unet()
    
    model = DeepPI(with_aux_output=False, use_leaky_relu=True, use_rgb_conv=False, use_disks=True, with_prev_mask=False)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.unet = deepcopy(unet)
    model.to(device)

    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    return model