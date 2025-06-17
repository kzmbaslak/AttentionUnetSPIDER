# src/inference.py
import torch
import SimpleITK as sitk
import numpy as np
from model3d import AttentionUnet3D
import os
import config
import cv2
from LightweightAttentionUnet3D import LightweightAttentionUnet3D
import matplotlib.pyplot as plt


# Modelin yükleneceği cihazı ayarla
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, num_classes):
    """Eğitilmiş modeli yükle"""
    model = LightweightAttentionUnet3D(in_channels=1, out_channels=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)["state_dict"])
    model.eval()  # inference moduna al
    return model

def predict_segmentation(image_path, model, num_classes, time_steps=5):
    """Verilen görüntü için segmentasyon tahmini yap"""
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image) # T, H, W

     # Zaman serisi uzunluğunu kontrol et ve gerekirse kırp veya doldur
    # T = image_array.shape[2]
    # if T > time_steps:
    #     start = (T - time_steps) // 2 #Ortala
    #     image_array = image_array[:,:,start:start + time_steps]
    # elif T < time_steps:
    #     pad_size = time_steps - T
    #     image_array = np.pad(image_array, ((0, pad_size), (0, 0), (0, 0)), mode='constant') # Zaman boyutunda doldur

    T = image_array.shape[0]
       
    if T > time_steps:
        start = (T - time_steps) // 2 #Ortala
        image_array = image_array[start:start + time_steps,:,:]
    elif T < time_steps:
        pad_size = time_steps - T
        image_array = np.pad(image_array, ((0, 0), (0, 0),(0, pad_size)), mode='constant') # Zaman boyutunda doldur
        
    #auto crop
    threshold_min = config.pixelMinValue # Alt yoğunluk eşiği
    threshold_max = config.pixelMaxValue # Üst yoğunluk eşiği
    image_array = automatic_roi(image_array,threshold_min, threshold_max)
    
    
    # Windowing
    image_array = windowing(image_array, config.pixelMinValue, config.pixelMaxValue)


    
    # Resize
    image_array = resizeImages(image_array)

    # Normalizasyon
    image_array = normalize(image_array)
    
    
    
    image_array = np.transpose(image_array, (2, 0, 1))
    #print("mask_array unique value: ",np.unique(mask_array))
    
        
    #plt.figure()
    #plt.imshow(mask_array[0],cmap="gray")
    
    
    # PyTorch için format
    image_array = np.expand_dims(image_array, axis=0) # Kanal boyutunu ekle
    image_array = np.expand_dims(image_array, axis=0) # Kanal ve batch boyutlarını ekle *********
    image_tensor = torch.from_numpy(image_array).float().to(DEVICE)


    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).cpu().numpy() # En olası sınıfı seç

    return prediction[0]  # Batch boyutunu kaldır

def resizeImages(imageArray):
    #print(type(maskArray),maskArray.shape,type(imageArray),imageArray.shape)
    
    # tempImageArray = np.zeros((config.patch_shape[::-1]),np.int64)
    # for i in range(imageArray.shape[2]):
    #     tempImageArray[:,:,i] = cv2.resize(imageArray[:,:,i], config.patch_shape[1:])
    # return tempImageArray

    #kesitsiz
    tempImageArray = np.zeros((imageArray.shape[0],*config.patch_shape[1:]),np.int64)
    for i in range(imageArray.shape[0]):
        tempImageArray[i,:,:] = cv2.resize(imageArray[i,:,:], config.patch_shape[1:])
    return tempImageArray

def windowing(image, minValue, maxValue):
    image = image.astype(np.float32)
    image = np.clip(image, minValue, maxValue)
    return image

def resample(image, new_solution=[1.0, 1.0, 1.0]):
    """
    Görüntü ve maskeyi SimpleITK kullanarak yeniden örneklendirir.

    Args:
        image (sitk.Image): Yeniden örneklenecek SimpleITK görüntü nesnesi.
        mask (sitk.Image): Yeniden örneklenecek SimpleITK maske nesnesi.
        new_solution (list): Hedef voksel aralığı (mm cinsinden).

    Returns:
        tuple: Yeniden örneklendirilmiş görüntü ve maske verilerini içeren NumPy dizileri.
    """

    # 1. Orijinal voksel aralığını al
    original_spacing = image.GetSpacing()

    # 2. Orijinal boyutu al
    original_size = image.GetSize()
    #print("original_spacing:",original_spacing,"  originalSize:",original_size)
    # 3. Hedef boyutu hesapla
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_solution[i]))) for i in range(image.GetDimension())]
    #print("newSize:",new_size)
    # 4. Yeniden örnekleme filtresini oluştur
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_solution)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetSize(new_size) # Hedef boyutu ayarla
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    # 5. Görüntüyü yeniden örnekle (lineer interpolasyon)
    resample.SetInterpolator(sitk.sitkLinear)
    new_img = resample.Execute(image)

    # 6. Maskeyi yeniden örnekle (en yakın komşu interpolasyonu)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # 7. Sonuçları NumPy dizilerine dönüştür
    img_array = sitk.GetArrayFromImage(new_img)

    return img_array


def normalize(image):
    image = (image - config.pixelMinValue) / (config.pixelMaxValue - config.pixelMinValue)
    #image[image > 1] = 1.
    #image[image < 0] = 0.
    return image


def automatic_roi(image_array, threshold_min, threshold_max):
    """Belirli bir yoğunluk aralığındaki pikselleri içeren ROI'yi belirler."""
    mask = (image_array > threshold_min) & (image_array < threshold_max)
    # Maskeyi kullanarak ROI'yi kırpma (sadece içeriği olan bölgeleri)
    indices = np.where(mask) # maskelenen yerlerin indexlerini al
    #print("indices: ",indices)
    if indices[0].size == 0: # eğer hiçbir index bulunamazsa
        print("Hata: Verilen yoğunluk aralığında piksel bulunamadı.")
        return None, None
    z_min, z_max = np.min(indices[0]), np.max(indices[0])
    y_min, y_max = np.min(indices[1]), np.max(indices[1])
    x_min, x_max = np.min(indices[2]), np.max(indices[2])

    roi = np.array(image_array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1])
    #return roi, (z_min, y_min, x_min, z_max+1, y_max+1, x_max+1) #+1 leri aralıklar dahil edilsin diye ekledim
    
    
    return roi

# Örnek Kullanım
if __name__ == '__main__':
    # Modelin yolu ve sınıf sayısı
    MODEL_PATH = 'saved_models/attention_unet 100 epoch LW.pth'
    NUM_CLASSES = 4
    TIME_STEPS = 5
    # Test görüntüsünün yolu
    TEST_IMAGE_PATH = 'data/images/155_t1.mha'  # Test için bir görüntü seç

    # Modeli yükle
    model = load_model(MODEL_PATH, NUM_CLASSES)

    # Segmentasyon tahmini yap
    segmented_image = predict_segmentation(TEST_IMAGE_PATH, model, NUM_CLASSES, TIME_STEPS)

    # Sonucu kaydet (isteğe bağlı)
    output_path = 'segmented_image1.mha'
    segmented_image = sitk.GetImageFromArray(segmented_image)
    sitk.WriteImage(segmented_image, output_path)

    print(f"Segmented image saved to {output_path}")
    predImage = sitk.ReadImage('segmented_image.mha')
    predArray = sitk.GetArrayFromImage(predImage)
    for i in np.arange(5):
        plt.figure()
        plt.imshow(predArray[i])
    