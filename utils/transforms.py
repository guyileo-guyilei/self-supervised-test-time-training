import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import JPEG
from torchvision.io import decode_jpeg

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0, std=0.25):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    

class ImpulseNoise(torch.nn.Module):
    def __init__(self, intensity=0.3):
        super().__init__()
        self.intensity = intensity

    def forward(self, tensor):
        noise_mask = torch.rand_like(tensor)  # Random values in range [0,1]

        p_dark = self.intensity / 2
        p_light = 1 - self.intensity / 2
        
        light_mask = noise_mask > p_light
        dark_mask = noise_mask < p_dark

        output = tensor.clone()
        output[light_mask] = 1.0
        output[dark_mask] = 0.0
        return output
    
    
class Pixelate(torch.nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        
    def forward(self, tensor):
        tensor_shape = tensor.shape
        if len(tensor_shape) < 4:
            tensor = tensor.unsqueeze(0)
        h = tensor_shape[-2]
        w = tensor_shape[-1]

        downscaled = F.interpolate(tensor, size=(h//self.scale, w//self.scale), mode="nearest")
        if len(downscaled.shape) < 4:
            downscaled = downscaled.unsqueeze(0)
        upscaled = F.interpolate(downscaled, size=(h, w), mode="nearest")
        return upscaled
        
        
class Defocus(torch.nn.Module):
    def __init__(self, kernel_size=11):
        super().__init__()
        self.blur = transforms.GaussianBlur(kernel_size)
        
    def forward(self, tensor):
        return self.blur(tensor)
    
        
class JPEGCompression(torch.nn.Module):
    def __init__(self, quality=10):
        super().__init__()
        self.compress = JPEG(quality)
        
    def forward(self, tensor):
        device = tensor.device
        dtype = tensor.dtype
        tensor = tensor.cpu().clamp(0, 1)
        tensor = (tensor*255).to(torch.uint8)
        return self.compress(tensor).to(device=device,dtype=dtype) / 255


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, tensor):
        return tensor
