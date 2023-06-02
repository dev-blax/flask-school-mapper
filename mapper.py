import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet18(pretrained='cifar10')

model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = 'static/dog.jpg'

image = Image.open(image_path)

image = transform(image)

image = image.unsqueeze(0)

with torch.no_grad():
    outputs = model(image)

_, predicted = torch.max(outputs, 1)

print(predicted.item())
