import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def main():
    start_time = time.time()
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Configuration
    data_dir = r"e:\projects\agri new\train_images"
    num_classes = 10
    batch_size = 32
    num_epochs = 3
    learning_rate = 0.001

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    
    # 80/20 train-val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training on {train_size} images, validating on {val_size} images.")

    # Initialize MobileNetV2
    print("Initializing MobileNetV2...")
    model = models.mobilenet_v2(pretrained=True)
    # Modify classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0.0
    val_metrics = {}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / train_size
        print(f"Train Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / val_size
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {acc:.4f} F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            cm = confusion_matrix(all_labels, all_preds)
            val_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm.tolist()
            }
            # Save model
            os.makedirs(r"e:\projects\agri new\model_weights", exist_ok=True)
            torch.save(model.state_dict(), r"e:\projects\agri new\model_weights\mobilenet_v2_paddy.pt")
            
    print("Training complete.")
    end_time = time.time()
    
    val_metrics["training_time_seconds"] = end_time - start_time
    
    with open(r"e:\projects\agri new\real_world_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=4)
        
    print("Metrics saved to real_world_metrics.json")

if __name__ == "__main__":
    main()
