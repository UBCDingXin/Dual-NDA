from .autoencoder import *
from .CcGAN_SAGAN import CcGAN_SAGAN_Generator, CcGAN_SAGAN_Discriminator
from .ResNet_embed import ResNet18_embed, ResNet34_embed, ResNet50_embed, model_y2h
from .ResNet_regre_eval import ResNet34_regre_eval
from .ResNet_class_eval import ResNet34_class_eval

from .vgg import vgg8, vgg11, vgg13, vgg16, vgg19
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

cnn_dict = {
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'vgg8': vgg8,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
}