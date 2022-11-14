from pathlib import Path
from typing import Dict
from datetime import datetime
import json

import torch
from torchvision import transforms
from torch import nn

import models

# transformations

ResNet50m_img_transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Lambda(lambda img : img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])


# loss
# https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss

MARGIN = 0.2 # Sketching without Worrying

# default distance function for triplet margin loss (eventually the dimension has to be adapted)
def euclidean_distance(t1:torch.Tensor, t2:torch.Tensor) -> float:
    print(torch.sum( torch.pow(t2 - t1, 2), dim=1))
    return torch.sqrt(torch.sum( torch.pow(t2 - t1, 2), dim=2))

#triplet_euclidean_loss = nn.TripletMarginWithDistanceLoss(margin=MARGIN, distance_function=euclidean_distance)

triplet_euclidean_loss = nn.TripletMarginLoss(margin=MARGIN)

# model saver and loader

# loads resnet50m state dicts or arbitrary models
def load_model(name:str) -> nn.Module:
    path = Path("models/") / name
    loaded = torch.load(path)
    model = None

    if isinstance(loaded, dict):
        print("Dictionary used to load model")
        model = models.ModifiedResNet(layers=(3, 4, 6, 3), output_dim=1024, heads=4, input_resolution=224, width=64)
        model.load_state_dict(loaded)
    else:
        print("Model completely loaded from file")
        model = loaded

    print(f"Model {name} loaded")
    return model

def save_model(model:nn.Module, data_dict:Dict, training_dict:Dict={}, param_dict:Dict={}, inference_dict:Dict={}) -> None:
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_name = f"{model.__class__.__name__}_{data_dict['dataset']}_{date_time}"
    suffix = "pth"
    model_path = Path("models") / f"{model_name}.{suffix}"

    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_name}.{suffix}")

    result_path = Path("results")
    if not result_path.is_dir(): result_path.mkdir(parents=True, exist_ok=True)

    result_path = result_path / model_name

    result_path.mkdir(parents=True, exist_ok=True)

    with open(result_path / "data_params.json", "w") as f:
        json.dump(data_dict, f, indent=4)

    with open(result_path / "training.json", "w") as f:
        json.dump(training_dict, f, indent=4)

    with open(result_path / "training_params.json", "w") as f:
        json.dump(param_dict, f, indent=4)

    with open(result_path / "inference.json", "w") as f:
        json.dump(inference_dict, f, indent=4)

    print(f"Data saved in {str(result_path)}")

model = load_model("ResNet50m.pth")
save_model(model, {'dataset':"test"})
#model2 = load_model("CLIP_ResNet-50.pt")
