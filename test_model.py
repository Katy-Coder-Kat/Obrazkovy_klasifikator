import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

model_path = "data/models/model_weights.pth"
image_path = r"C:\Users\ranoc\OneDrive\Desktop\obrazkovy editor\data\raw\raw-img\cane\test\test_image.jpg"
class_names = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_image(model, image_path, class_names, device):
    image = preprocess_image(image_path).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

print(f"Načítám obrázek z cesty: {image_path}")
prediction = predict_image(model, image_path, class_names, device)
print(f"Předpověď: {prediction}")
