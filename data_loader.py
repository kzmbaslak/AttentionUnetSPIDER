
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
import cv2
import matplotlib.pyplot as plt
from collections import Counter


class SpineDataset(Dataset):
    """
    mask_array için 
    100 => kanal
    201,202,203,204,205,206,207,208,209 => omurlar arası disik
    
    """
    def __init__(self, image_dir, mask_dir, dataframe, transform=None, subset='training', test_size=0.16, time_steps=config.patch_shape[0]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dataframe = dataframe
        self.transform = transform
        self.subset = subset
        self.image_names = self.dataframe['filename'].tolist()
        self.time_steps = time_steps # Time step sayısı

        # Train/Validation/Test split'i overview.csv'den oku
        if subset == 'training' or subset == 'validation':
            self.image_names = self.dataframe[self.dataframe['subset'] == subset]['filename'].tolist()
            if subset == 'training':
                self.image_names, test_names = train_test_split(self.image_names, test_size=test_size, random_state=42)
        elif subset == 'test':
            # Test verisi overview'da yoksa, training'den ayır
            train_df = self.dataframe[self.dataframe['subset'] == 'training']
            train_names = train_df['filename'].tolist()
            train_names, test_names = train_test_split(train_names, test_size=test_size, random_state=42)
            self.image_names = test_names
        else:
            raise ValueError("Subset training, validation veya test olmalı.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        #print(f"[Dataset] __getitem__ çağrıldı: {idx}")
        image_name = self.image_names[idx]
        image_path = self.image_dir+ "/"+ image_name+".mha"
        mask_path = self.mask_dir+ "/" +image_name+".mha"
        
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # Resampling
        image, mask = sitk.ReadImage(image_path), sitk.ReadImage(mask_path) #Resample fonksiyonu için okuma işlemi
        image_array, mask_array = self.resample(image, mask, config.working_resolution)

        
        if(image_array.shape != mask_array.shape):
            return
        
        #Ters dizi
        # Zaman serisi uzunluğunu kontrol et ve gerekirse kırp veya doldur
        T = image_array.shape[2]
        
        if T > self.time_steps:
            start = (T - self.time_steps) // 2 #Ortala
            image_array = image_array[:,:,start:start + self.time_steps]
            mask_array = mask_array[:,:,start:start + self.time_steps]
        elif T < self.time_steps:
            pad_size = self.time_steps - T
            image_array = np.pad(image_array, ((0, pad_size), (0, 0), (0, 0)), mode='constant') # Zaman boyutunda doldur
            mask_array = np.pad(mask_array, ((0, pad_size), (0, 0), (0, 0)), mode='constant') # Zaman boyutunda doldur
        
       
        
        
        #auto crop
        threshold_min = config.pixelMinValue # Alt yoğunluk eşiği
        threshold_max = config.pixelMaxValue # Üst yoğunluk eşiği
        image_array = self.automatic_roi(image_array,threshold_min, threshold_max)
        mask_array = self.automatic_roi(mask_array,threshold_min, threshold_max)
        
        # Windowing
        image_array = self.windowing(image_array, config.pixelMinValue, config.pixelMaxValue)


        
        # Resize
        image_array, mask_array = self.resizeImages(image_array, mask_array)

        # Normalizasyon
        image_array = self.normalize(image_array)
        
        
        
        image_array = np.transpose(image_array, (2, 0, 1))
        mask_array = np.transpose(mask_array, (2, 0, 1))
        #print("mask_array unique value: ",np.unique(mask_array))
        
        mask_array[(mask_array > 0) & (mask_array < 80)] = 1
        mask_array[mask_array == 100] = 2
        mask_array[mask_array > 200] = 3
        
            
        #plt.figure()
        #plt.imshow(mask_array[0],cmap="gray")
        
        
        # PyTorch için format
        image_array = np.expand_dims(image_array, axis=0) # Kanal boyutunu ekle
        mask_array = mask_array.astype(np.int64)  # CrossEntropyLoss için long tipinde olmalı

        image_array = torch.from_numpy(image_array).float()
        mask_array = torch.from_numpy(mask_array).long() # Maskeler long tipinde
        #print("Mask sinıfları:", torch.unique(mask_array))
        #assert mask_array.dtype == torch.long

        if self.transform:
            image_array, mask_array = self.transform(image_array, mask_array)
            

        return image_array, mask_array

    def resizeImages(self, imageArray, maskArray):
        #print(type(maskArray),maskArray.shape,type(imageArray),imageArray.shape)
        
        #kesitli ayrıca ters dizi
        tempImageArray = np.zeros((config.patch_shape[::-1]),np.int64)
        tempMaskArray = np.zeros((config.patch_shape[::-1]),np.int64)
        
        for i in range(imageArray.shape[2]):
            tempImageArray[:,:,i] = cv2.resize(imageArray[:,:,i], config.patch_shape[1:])
            tempMaskArray[:,:,i] = cv2.resize(maskArray[:,:,i], config.patch_shape[1:],interpolation=cv2.INTER_NEAREST)
        return tempImageArray, tempMaskArray
    
    def windowing(self, image, minValue, maxValue):
        image = image.astype(np.float32)
        image = np.clip(image, minValue, maxValue)
        return image

    def resample(self, image, mask, new_solution=[1.0, 1.0, 1.0]):
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
        new_mask = resample.Execute(mask)
    
        # 7. Sonuçları NumPy dizilerine dönüştür
        img_array = sitk.GetArrayFromImage(new_img)
        mask_array = sitk.GetArrayFromImage(new_mask)
    
        return img_array, mask_array


    def normalize(self, image):
        image = (image - config.pixelMinValue) / (config.pixelMaxValue - config.pixelMinValue)
        #image[image > 1] = 1.
        #image[image < 0] = 0.
        return image
    
    
    def automatic_roi(self, image_array, threshold_min, threshold_max):
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


def analyze_mha_files(directory):
    """
    Verilen dizindeki tüm MHA dosyalarını analiz eder ve her bir dosyadaki
    sayısal değerlerin frekansını ekrana yazdırır.

    Args:
        directory (str): MHA dosyalarının bulunduğu dizin.
    """

    for filename in os.listdir(directory):
        if filename.endswith(".mha"):
            
            filepath = directory+ "/"+ filename
            try:
                # MHA dosyasını oku
                image = sitk.ReadImage(filepath)
                image_array = sitk.GetArrayFromImage(image)

                # Dizideki sayısal değerlerin frekansını hesapla
                value_counts = Counter(image_array.flatten())

                # Sonuçları ekrana yazdır
                print(f"Dosya: {filename}")
                for value, count in sorted(value_counts.items()):
                    print(f"  Değer: {value}, Frekans: {count}")
                print("-" * 30)  # Dosyalar arasında ayırıcı

            except Exception as e:
                print(f"Hata: {filepath} dosyası işlenirken bir hata oluştu: {e}")



if __name__ == '__main__':

#    directory_path = "data/masks"  # MHA dosyalarınızın bulunduğu dizini buraya girin
#    analyze_mha_files(directory_path)    


    import torch
    # Veri klasörleri ve CSV dosyasının yolu
    IMAGE_DIR = config.IMAGE_DIR
    MASK_DIR = config.MASK_DIR
    CSV_PATH = config.overviewPath
    TIME_STEPS = 5 #Zaman serisi boyutu


    # Dataframe'i oku
    df = pd.read_csv(CSV_PATH,sep=';')

    # Dataset'leri oluştur
    train_dataset = SpineDataset(IMAGE_DIR, MASK_DIR, df, subset='training', time_steps=TIME_STEPS)
    val_dataset = SpineDataset(IMAGE_DIR, MASK_DIR, df, subset='validation', time_steps=TIME_STEPS)
    test_dataset = SpineDataset(IMAGE_DIR, MASK_DIR, df, subset='test', time_steps=TIME_STEPS)
    print(len(train_dataset),len(val_dataset),len(test_dataset))
    
    # Dataloader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Bir batch al ve boyutları kontrol et
    for i, (images, masks) in enumerate(train_loader):
        print(f"Batch {i+1} - Image shape: {images.shape}, Mask shape: {masks.shape}")
        break
    