import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from model import AttentionUnet
from model3d import AttentionUnet3D
from data_loader import SpineDataset
from utils import dice_loss, dice_coefficient, calculate_metrics
import numpy as np
import SimpleITK as sitk
import config



# Hiperparametreler
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
NUM_CLASSES = 4 # Arka plan dahil
NUM_WORKERS = 4 # multiprocess sayısı
PIN_MEMORY = True # Adresleri sabitlenmiş hafıza  oluşturma
IMAGE_DIR = 'data/images'
MASK_DIR = 'data/masks'
CSV_PATH = config.overviewPath
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')
MODEL_SAVE_PATH = 'saved_models/attention_unet.pth'
TIME_STEPS = 5

# Veri yükleme
df = pd.read_csv(CSV_PATH,sep=';')
train_dataset = SpineDataset(IMAGE_DIR, MASK_DIR, df, subset='training', time_steps=TIME_STEPS)
val_dataset = SpineDataset(IMAGE_DIR, MASK_DIR, df, subset='validation', time_steps=TIME_STEPS)
test_dataset = SpineDataset(IMAGE_DIR, MASK_DIR, df, subset='test', time_steps=TIME_STEPS)  # Test dataset'i de oluşturuldu

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) # Test loader'ı da oluşturuldu

# Model, optimizer ve kayıp fonksiyonu
#model = AttentionUnet(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE) # in_channels = 1 (Gri tonlamalı)
model = AttentionUnet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE) # in_channels = 1 (Gri tonlamalı)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Eğitim döngüsü
def train_loop(model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:            
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            print("Model output shape:", outputs.shape)
            print("Target mask shape:", masks.shape)
            
            print("Output shape:", outputs.shape)
            print("Mask shape:", masks.shape)
            print("Mask dtype:", masks.dtype)
            print("Mask max sınıf:", masks.max().item())
            print("Mask sınıflar:", masks.unique().tolist())
            
            if masks.max() >= outputs.shape[1]:
                raise ValueError(f"Mask'te {masks.max().item()} sınıfı var ama modelin out_channels={outputs.shape[1]}")
            
            # Flatten for CrossEntropy
            outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, outputs.shape[1])
            masks = masks.view(-1)
            
            
            loss = criterion(outputs, masks)
            
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                print("🔥 CUDA RuntimeError:", e)
                torch.cuda.synchronize()
                raise
            
            
            #loss.backward()
            #optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_dice = validation_loop(model, criterion, val_loader, device)  # val_dice eklendi


        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        # Modeli kaydet (en iyi validation kaybıyla)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'Model saved to {MODEL_SAVE_PATH}')

def validation_loop(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    dice_scores = [] # Her batch için dice skorlarını sakla

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Dice Coefficient hesapla
            probs = torch.softmax(outputs, dim=1)
            dice = dice_coefficient(probs, masks, num_classes=NUM_CLASSES, device=device)  # num_classes eklendi
            dice_scores.append(dice.cpu().numpy())  # GPU'dan CPU'ya taşı ve NumPy'ye dönüştür

    val_loss /= len(val_loader)
    mean_dice = np.mean(dice_scores) # Tüm batch'lerin ortalama dice skoru
    return val_loss, mean_dice

# Eğitim
if __name__ == '__main__':
    train_loop(model, optimizer, criterion, train_loader, val_loader, NUM_EPOCHS, DEVICE)

    # Test performansı
    from utils import test_loop
    test_loop(model, test_loader, criterion, DEVICE, num_classes=NUM_CLASSES)