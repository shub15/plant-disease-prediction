"""
MobileNetV3-Large Plant Disease Training Script (Ready for Google Colab/Kaggle)
=============================================================================

Instructions for Google Colab:
1. Upload your dataset to Google Drive or Colab.
   - The folder structure MUST be:
     dataset/
       ├── Apple - Apple Scab/
       │   ├── img1.jpg
       │   └── img2.jpg
       ├── Apple - Healthy/
       ...
2. Go to 'Runtime' > 'Change runtime type' and select 'T4 GPU'.
3. Change the `DATA_DIR` variable below to point to your dataset folder.
4. Run this script. It will automatically save `best_mobilenetv3_large.pth`.
"""

import os
import time
import copy
import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "./plantvillage-dataset/raw/color"  # The color dataset from the repo
BATCH_SIZE = 32                                # Increase to 64 if you have more GPU VRAM
NUM_EPOCHS = 15                                # 15-20 epochs is usually enough for fine-tuning
LEARNING_RATE = 0.001
NUM_CLASSES = 38                               # PlantVillage has 38 classes

# ==========================================
# 1. DATA AUGMENTATION & LOADING
# ==========================================
# Heavy augmentation for training to improve real-world accuracy
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # Standard ImageNet normalization
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_data():
    if not os.path.exists(DATA_DIR):
        print("Dataset not found locally. Cloning from GitHub (this may take a few minutes)...")
        subprocess.run(["git", "clone", "https://github.com/spmohanty/plantvillage-dataset.git"], check=True)
        
    print(f"Loading dataset from: {DATA_DIR}")
    full_dataset = datasets.ImageFolder(DATA_DIR)
    
    # Split 80% Training / 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}
    class_names = full_dataset.classes
    
    print(f"Classes found: {len(class_names)}")
    print(f"Training images: {train_size}")
    print(f"Validation images: {val_size}")
    
    return dataloaders, dataset_sizes, class_names

# ==========================================
# 2. MODEL SETUP (MobileNetV3-Large)
# ==========================================
def build_model():
    print("Building MobileNetV3-Large model...")
    # Load pre-trained MobileNetV3-Large
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    
    # Freeze early layers (optional, but good for fast training)
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final classifier head
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Ensure the new classifier requires gradients
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            # Use tqdm for a progress bar
            loop = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}", leave=False)
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it has the best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'best_mobilenetv3_large.pth')
                print(f"🌟 New best model saved! (Accuracy: {best_acc:.4f})")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware used for training: {device}")

    dataloaders, dataset_sizes, class_names = load_data()
    model = build_model().to(device)

    # Use CrossEntropyLoss (Or FocalLoss if data is unbalanced)
    criterion = nn.CrossEntropyLoss()

    # Optimize only the classifier parameters since we froze the rest
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # Decay Learning Rate by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("\n🚀 Starting Training...")
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, device, num_epochs=NUM_EPOCHS)
    
    print("\n✅ Training Complete. You can download 'best_mobilenetv3_large.pth' and use it in your Raspberry Pi pipeline.")

if __name__ == '__main__':
    main()
