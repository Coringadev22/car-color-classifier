import os
import torch
import torchvision
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

# === Configurações ===
data_dir = 'data'
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_classes = len(os.listdir(os.path.join(data_dir, 'train')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformações ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Carregar datasets ===
train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Carregar modelo ResNet18 (com fallback se a versão do PyTorch for mais antiga) ===
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except:
    model = models.resnet18(pretrained=True)

# === Ajustar camada final para o número de classes ===
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# === Função de perda e otimizador ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Loop de treinamento ===
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / len(train_loader))

    print(f"Final da época [{epoch+1}/{num_epochs}], Loss médio: {total_loss/len(train_loader):.4f}")

# === Avaliação no conjunto de validação ===
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nRelatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# === Salvar modelo ===
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes
}, 'car_color_model.pt')

print("\n✅ Modelo salvo como car_color_model.pt")
