import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from model3d import AttentionUnet3D
from LightweightAttentionUnet3D import LightweightAttentionUnet3D
from data_loader import SpineDataset
from utils import dice_loss, dice_coefficient, calculate_metrics, compute_metrics, saveValLoss, saveAllEpochMetrics
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import config
import matplotlib.pyplot as plt



# Hiperparametreler
LOAD_MODEL = True
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
NUM_CLASSES = 4 # Arka plan dahil
NUM_WORKERS = 2 # multiprocess sayısı
PIN_MEMORY = True # Adresleri sabitlenmiş hafıza  oluşturma
IMAGE_DIR = config.IMAGE_DIR
MASK_DIR = config.MASK_DIR
CSV_PATH = config.overviewPath
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')
MODEL_SAVE_PATH = 'saved_models/attention_unet.pth'
METRIC_SAVE_PATH = "saved_metrics/val_metrics.csv"
LOSS_SAVE_PATH = "saved_metrics/val_loss.csv"
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
# model = AttentionUnet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE) # in_channels = 1 (Gri tonlamalı)
model = LightweightAttentionUnet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE) # in_channels = 1 (Gri tonlamalı)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Eğitim döngüsü
def train_loop(model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(MODEL_SAVE_PATH), model)
        val_loss, val_metrics = validation_loop(model, criterion, val_loader, device)
        best_val_loss = val_loss
    else:
        best_val_loss = float('inf')
    
    
    
    val_losses = []
    val_of_metrics_all_epoch = []
    train_losses = []
    
    if(os.path.exists(LOSS_SAVE_PATH)):
        val_losses = pd.read_csv(LOSS_SAVE_PATH).values.tolist()
        best_val_loss = np.min(val_losses)
    
    if(os.path.exists(METRIC_SAVE_PATH)):
        val_of_metrics_all_epoch = pd.read_csv(METRIC_SAVE_PATH).values.tolist()
        
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        
        for i, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # Flatten for CrossEntropy
            #outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, outputs.shape[1])
            #masks = masks.view(-1)
            
            
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
            train_losses.append(loss.item())

        train_loss /= len(train_loader)
        val_loss, val_metrics = validation_loop(model, criterion, val_loader, device)
        
        val_losses.append([val_loss])
        saveValLoss(val_losses)
        
        val_of_metrics_all_epoch.append(val_metrics)
        saveAllEpochMetrics(val_of_metrics_all_epoch)
        #print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Loss SD: {np.std(train_losses)}, Train Loss Mean: {np.mean(train_losses)}, Val Loss: {val_loss:.4f}, Val Dice: {val_metrics[0]:.4f}')

        # Modeli kaydet (en iyi validation kaybıyla)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f'Model saved to {MODEL_SAVE_PATH}')
    
    
    
    #print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
    
    #pd.DataFrame(val_of_metrics_all_epoch).plot(title="metrics of validation for All epochs")
    #plt.show()
    #plt.figure()
    #plt.title("Validation Loss")
    #plt.plot(val_losses)

def validation_loop(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    #dice_scores = [] # Her batch için dice skorlarını sakla
    metrics = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Dice Coefficient hesapla
            #probs = torch.softmax(outputs, dim=1)
            #dice = dice_coefficient(probs, masks, num_classes=NUM_CLASSES, device=device)  # num_classes eklendi

            metrics.append(compute_metrics(masks,outputs))
            
            #dice_scores.append(dice.cpu().numpy())  # GPU'dan CPU'ya taşı ve NumPy'ye dönüştür

    df_metrics = pd.DataFrame(metrics)
    df_metrics.plot(title="metrics of validation")
    plt.show()
   
    val_loss /= len(val_loader)
    #mean_dice = np.mean(dice_scores) # Tüm batch'lerin ortalama dice skoru
    
    #plt.figure()
    #plt.title("Validation dice scores")
    #plt.plot(dice_scores)
    return val_loss, df_metrics.mean()

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    print("=> finished load porcess")
# Eğitim
if __name__ == '__main__':
    train_loop(model, optimizer, criterion, train_loader, val_loader, NUM_EPOCHS, DEVICE)

    # Test performansı
    from utils import test_loop
    test_loop(model, test_loader, criterion, DEVICE, num_classes=NUM_CLASSES)