import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from PIL import ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    print("🚀 Training started...")

    DATA_DIR = 'dataset/dataset_blood_group'
    MODEL_SAVE_PATH = 'fingerprint_blood_group_model.pth'

    # 🔁 Advanced Data Augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 📁 Dataset loading
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f"✅ Found {len(train_dataset)} images across {len(train_dataset.classes)} classes: {train_dataset.classes}")

    # 🧠 Load Pretrained ResNet18 & Unfreeze All Layers
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ✅ Loss Function + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # 🔁 Training Loop
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"📊 Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # 💾 Save Trained Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved at '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()
