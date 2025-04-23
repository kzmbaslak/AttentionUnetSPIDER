# src/utils.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import SimpleITK as sitk

def dice_loss(probs, target, epsilon=1e-6, num_classes=4):
    """Çok sınıflı Dice kaybı"""
    dice = 0
    for i in range(num_classes):
        p = probs[:, i, :, :, :].contiguous().view(-1)
        t = (target == i).float().contiguous().view(-1)
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice += (2 * intersection + epsilon) / (union + epsilon)

    return 1 - dice / num_classes #Ortalamasını al

def dice_coefficient(probs, target, epsilon=1e-6, num_classes=4, device='cuda'):
    """Çok sınıflı Dice katsayısı"""
    dice = 0
    for i in range(num_classes):
        p = probs[:, i, :, :, :].contiguous().view(-1)
        t = (target == i).float().contiguous().view(-1)
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice += (2 * intersection + epsilon) / (union + epsilon)

    return dice / num_classes #Ortalamasını al

def calculate_metrics(outputs, targets, num_classes):
    # İstenilen metrikleri hesapla (örneğin, Dice, IoU, Precision, Recall)
    pass  # Implemente edilecek

def test_loop(model, test_loader, criterion, device, num_classes):
    model.eval()  # evaluation moduna al
    test_loss = 0.0
    dice_scores = []

    with torch.no_grad():  # gradyan hesaplamayı kapat
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images) # Tahminleri al
            loss = criterion(outputs, masks) # Kaybı hesapla
            test_loss += loss.item()

            # Dice Coefficient hesapla
            probs = torch.softmax(outputs, dim=1)
            dice = dice_coefficient(probs, masks, num_classes=num_classes, device=device)
            dice_scores.append(dice.cpu().numpy())

    test_loss /= len(test_loader)
    mean_dice = np.mean(dice_scores)
    print(f'Test Loss: {test_loss:.4f}, Test Dice: {mean_dice:.4f}')

#Görselleştirme fonksiyonu
def visualize_predictions(model, data_loader, device, save_dir, num_samples=5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            if i >= num_samples:
                break

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted_masks = torch.argmax(outputs, dim=1) # Sınıf tahminleri

            # Görselleştirmeyi kaydet
            for j in range(images.size(0)):  # Batch içindeki her örnek için
                image = images[j].cpu().numpy().squeeze()  # (C, H, W, D) -> (H, W, D)
                true_mask = masks[j].cpu().numpy()       # (H, W, D)
                pred_mask = predicted_masks[j].cpu().numpy()  # (H, W, D)

                # Verileri SimpleITK görüntü formatına dönüştür
                image_sitk = sitk.GetImageFromArray(image)
                true_mask_sitk = sitk.GetImageFromArray(true_mask.astype(np.uint8))  # Maskeler uint8 olmalı
                pred_mask_sitk = sitk.GetImageFromArray(pred_mask.astype(np.uint8))

                # Kayıt yolları
                image_path = os.path.join(save_dir, f'sample_{i}_{j}_image.mha')
                true_mask_path = os.path.join(save_dir, f'sample_{i}_{j}_true_mask.mha')
                pred_mask_path = os.path.join(save_dir, f'sample_{i}_{j}_pred_mask.mha')

                # Kaydet
                sitk.WriteImage(image_sitk, image_path)
                sitk.WriteImage(true_mask_sitk, true_mask_path)
                sitk.WriteImage(pred_mask_sitk, pred_mask_path)

                print(f'Sample {i}_{j} saved to {save_dir}')