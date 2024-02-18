import torch
from torch import nn
import torchvision

# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768
                 ):
        super().__init__()

