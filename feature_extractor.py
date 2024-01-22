import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, original_model, extract_layers=False):
        super(ResNet50FeatureExtractor, self).__init__()
        if extract_layers:
            self.features = list(original_model.children())[:-1]
        else:
            self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        if isinstance(self.features, nn.Sequential):
            return self.features(x)
        else:
            results = []
            for model in self.features:
                x = model(x)
                results.append(x)
            return results

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def preprocess_image(img_path, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0).to(device)
    return image
