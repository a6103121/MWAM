import torch
import torch.nn as nn
import torch_dct as dct
import numpy as np

class FrequencyLayer(nn.Module):
    def __init__(self, batch_size, input_size, patch_size=8,micro_patch_window_size=2):
        super(FrequencyLayer, self).__init__()
        self.input_H, self.input_W, self.channel = input_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.patch_Num_H = self.input_H // self.patch_size
        self.patch_Num_W = self.input_W // self.patch_size

        self.micro_patch_window_size = micro_patch_window_size
        self.micro_patch_num_H = self.patch_Num_H * self.micro_patch_window_size
        self.micro_patch_num_W = self.patch_Num_W * self.micro_patch_window_size

    def forward(self, input):
        patches = self.get_patches(input)

        # Apply 2D DCT
        patches = dct.dct_2d(patches, 'ortho')  # Assuming dct is a module with dct_2d function

        # Create masks for low and high frequency components
        low_mask = torch.zeros_like(patches, dtype=torch.bool, device=input.device)
        low_mask[:, :, :, :, 0:self.micro_patch_window_size, 0:self.micro_patch_window_size] = True

        high_mask = torch.zeros_like(patches, dtype=torch.bool, device=input.device)
        high_mask[:, :, :, :, (self.patch_size - self.micro_patch_window_size):self.patch_size,
                  (self.patch_size - self.micro_patch_window_size):self.patch_size] = True

        # Extract low and high frequency components
        low = patches[low_mask].view(self.batch_size,
                                     self.channel,
                                     self.patch_Num_H,
                                     self.patch_Num_W,
                                     self.micro_patch_window_size,
                                     self.micro_patch_window_size) \
            .permute(0, 2, 3, 1, 4, 5)

        high = patches[high_mask].view(self.batch_size,
                                       self.channel,
                                       self.patch_Num_H,
                                       self.patch_Num_W,
                                       self.micro_patch_window_size,
                                       self.micro_patch_window_size) \
            .flip(4).flip(5).permute(0, 2, 3, 1, 4, 5)

        return low, high

    def get_patches(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, self.patch_Num_H, self.patch_size, self.patch_Num_W, self.patch_size)
        y = x.permute(0, 2, 4, 1, 3, 5)
        return y

