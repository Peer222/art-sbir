import torch
from torchvision import transforms

ResNet50m_img_transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Lambda(lambda img : img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

