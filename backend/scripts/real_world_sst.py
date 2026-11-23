import os
import sys
import json
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.image_processing import AgriImageProcessor
from scripts.publication_prep import SSTEnsemble

class NGVCRealDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def main():
    print("=" * 70)
    print("  AgriPulse — Real-World SST Ensemble Validation (Part 2 Paper)")
    print("=" * 70)

    device = torch.device("cpu")
    data_dir = r"e:\projects\agri new\train_images"
    cache_file = r"e:\projects\agri new\real_world_sst_features.pt"
    
    num_classes = 10
    batch_size = 64
    num_epochs = 40
    learning_rate = 1e-3

    processor = AgriImageProcessor()

    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        cache = torch.load(cache_file)
        X = cache["X"]
        y = cache["y"]
        class_names = cache["classes"]
    else:
        print("Extracting NGVC features from dataset... This will take a few minutes.")
        X_list = []
        y_list = []
        
        classes = sorted(os.listdir(data_dir))
        class_names = classes
        
        start_time = time.time()
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
                
            img_names = os.listdir(cls_dir)
            print(f"  Processing {cls_name} ({len(img_names)} images)...")
            for img_name in img_names:
                img_path = os.path.join(cls_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract 128-bin NGVC
                ngvc = processor.extract_ngvc(img_rgb, bins=128)
                
                X_list.append(ngvc)
                y_list.append(cls_idx)
                
        print(f"Extraction complete in {time.time() - start_time:.1f}s")
        X = torch.tensor(np.array(X_list, dtype=np.float32))
        y = torch.tensor(np.array(y_list, dtype=np.int64))
        
        print(f"Saving to cache: {cache_file}")
        torch.save({"X": X, "y": y, "classes": class_names}, cache_file)

    print(f"Total dataset size: {len(X)}")
    
    # Create dataset and split
    full_dataset = NGVCRealDataset(X, y)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training on {train_size} images, validating on {val_size} images.")

    print("\nInitializing SSTEnsemble...")
    model = SSTEnsemble(bins=128, embed=64, num_classes=num_classes)
    model = model.to(device)

    # We only train classification, so severity loss is ignored
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0.0
    val_metrics = {}

    start_train_time = time.time()
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)  # ignore severity output
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / train_size

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / val_size
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

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
            # Save best model
            os.makedirs(r"e:\projects\agri new\model_weights", exist_ok=True)
            torch.save(model.state_dict(), r"e:\projects\agri new\model_weights\sst_ensemble_paddy.pt")

    print("\nTraining complete.")
    val_metrics["training_time_seconds"] = time.time() - start_train_time
    
    out_file = r"e:\projects\agri new\real_world_sst_metrics.json"
    with open(out_file, "w") as f:
        json.dump(val_metrics, f, indent=4)
        
    print(f"Best Validation Accuracy: {val_metrics['accuracy']*100:.2f}%")
    print(f"Best Macro F1-Score: {val_metrics['f1_score']:.4f}")
    print(f"Metrics saved to {out_file}")

if __name__ == "__main__":
    main()
