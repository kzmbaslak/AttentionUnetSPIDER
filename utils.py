# src/utils.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import config


METRIC_SAVE_PATH = "saved_metrics/val_metrics.csv"
LOSS_SAVE_PATH = "saved_metrics/val_loss.csv"


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
    metrics = []

    with torch.no_grad():  # gradyan hesaplamayı kapat
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images) # Tahminleri al
            loss = criterion(outputs, masks) # Kaybı hesapla
            test_loss += loss.item()

            # Dice Coefficient hesapla
            probs = torch.softmax(outputs, dim=1)
            dice = dice_coefficient(probs, masks, num_classes=num_classes, device=device)
            
            metrics.append(compute_metrics(masks,outputs))
            dice_scores.append(dice.cpu().numpy())

    test_loss /= len(test_loader)
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.plot(title="metrics of test")
    plt.show()
    
    mean_dice = np.mean(dice_scores)
    #plt.figure()
    #plt.title("test dice score")
    #plt.plot(dice_scores)
    
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
                
                
                
######################################################################################metrics


import pandas as pd
import torch

# ----------------------------------------
# 1. .mha dosyalarını oku (Mask ve Prediction)
# ----------------------------------------



# -------------------------
# 2. Metrik hesaplama
# -------------------------

def compute_metrics_for_class(mask, pred, cls):
    mask_cls = (mask == cls).type(torch.uint8)
    pred_cls = (pred == cls).type(torch.uint8)

    TP = torch.logical_and(pred_cls == 1, mask_cls == 1).sum()
    FP = torch.logical_and(pred_cls == 1, mask_cls == 0).sum()
    FN = torch.logical_and(pred_cls == 0, mask_cls == 1).sum()
    TN = torch.logical_and(pred_cls == 0, mask_cls == 0).sum()
    
    
    # Convert to float for division
    TP = TP.to(torch.float)
    FP = FP.to(torch.float)
    FN = FN.to(torch.float)
    TN = TN.to(torch.float)
    
    
    eps = 1e-7  # sıfıra bölünmeyi önlemek için

    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)

    #return dice, iou, precision, recall, f1, accuracy
    # Return CPU float values
    return (dice.item(), iou.item(), precision.item(), recall.item(), f1.item(), accuracy.item())



# -------------------------
# 3. Tüm sınıflar için hesapla
# -------------------------

def compute_metrics(masks,preds):
    preds = torch.argmax(preds,dim=1)
    macro_avg = []
    for mask, pred in zip(masks,preds):
        if preds.dim() == 5 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        assert mask.shape == pred.shape, "Shape mismatch!"
        assert mask.device == pred.device, "Mask and prediction must be on the same device!"

        results = []
        for cls in np.arange(config.classNumber):
            dice, iou, precision, recall, f1, acc = compute_metrics_for_class(mask, pred, cls)
            results.append({
                "Class": cls,
                "Dice": dice,
                "IoU": iou,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Accuracy": acc
            })
        df = pd.DataFrame(results)
        df.set_index("Class", inplace=True)
        #print("\n📊 Sınıf Bazlı Metrikler")
        #print(df.round(4))

        # -------------------------
        # 4. Ortalamalar
        # -------------------------

        macro_avg.append(df.mean(numeric_only=True))
        #print("\n🔢 Macro Ortalama Metrikler:")
        #print(macro_avg[0].round(4))
    macro_avg = pd.DataFrame(macro_avg).mean()
    #dice, iou, precision, recall, f1, acc
    return macro_avg
        
def saveValLoss(val_losses):
    val_losses_temp = pd.DataFrame(val_losses,columns=["loss"])
    #val_losses_temp.columns = ["loss"]
    val_losses_temp.to_csv(LOSS_SAVE_PATH,index=False)

def saveAllEpochMetrics(val_of_metrics_all_epoch):
    val_of_metrics_all_epoch_temp = pd.DataFrame(val_of_metrics_all_epoch)
    val_of_metrics_all_epoch_temp.columns = ["Dice","IoU","Precision","Recall","F1 Score","Accuracy"]
    val_of_metrics_all_epoch_temp.to_csv(METRIC_SAVE_PATH,index = False)
    