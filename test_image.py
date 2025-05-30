import torch
from torchvision import models, transforms
from PIL import Image
import os

# === Configurações ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'car_color_model.pt'
image_path = "data/test/red/5bbb7d8bc4_jpg.rf.0df0f0330fac6d3e69d9ca8b412bb217.jpg"

# === Carregar classes dinamicamente ===
classes = sorted(os.listdir('data/train'))
num_classes = len(classes)

# === Transformações === 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Carregar imagem ===
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# === Carregar modelo ===
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Carregar pesos salvos
checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval().to(device)

# === Fazer previsão ===
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = classes[predicted.item()]

# === Resultado ===
print(f'\n🟢 Cor prevista: {predicted_class.upper()}')
