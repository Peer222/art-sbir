"""
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("vincent-van-gogh-gross-jpg--18879-.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a person", "a painting", "a painting of Vincent van Gogh"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
"""

import torch
from PIL import Image
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"

#import clip
#available models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
#model, preprocess = clip.load("RN50", device=device)
#print(model)


#extracts visualTransformer from full clip model
#model = model.get_submodule(target="visual")

#torch.save(model, "../models/CLIP_ResNet-50.pt")


model = torch.load("../models/CLIP_ResNet-50.pt")
preprocess = torch.load("./Image_Preprocessing")

#object_methods = [method_name for method_name in dir(model) if callable(getattr(model, method_name))]

#print(object_methods)

#print(model.get_submodule(target="visual"))

images = torch.cat([preprocess(Image.open(f)).unsqueeze(0).to(device) for f in glob.iglob("./test_images/*")])

names = [f for f in glob.iglob("./test_images/*")]

sketch = preprocess(Image.open("./test_sketches/n02691156_58-1.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model(images)
    sketch_features = model(sketch)

    sketch_features /= sketch_features.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * sketch_features @ image_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(8)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{index}->{names[index]}: {100 * value.item():.2f}%")
