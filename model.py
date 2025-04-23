# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(out_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionUnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AttentionUnet, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.dec4 = DoubleConv(512, 256)
        self.dec3 = DoubleConv(256 + 256, 128)
        self.dec2 = DoubleConv(128 + 128, 64)
        self.dec1 = DoubleConv(64 + 64, num_classes)

        self.attn4 = AttentionBlock(in_channels=256, out_channels=512)
        self.attn3 = AttentionBlock(in_channels=128, out_channels=256)
        self.attn2 = AttentionBlock(in_channels=64, out_channels=128)


        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        self.final_conv = nn.Conv3d(num_classes, num_classes, kernel_size=1)


    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Decoder with Attention
        upconv4 = self.upconv4(enc4)
        attn4 = self.attn4(g=upconv4, x=enc3)
        dec4 = self.dec4(torch.cat((attn4, upconv4), dim=1))

        upconv3 = self.upconv3(dec4)
        attn3 = self.attn3(g=upconv3, x=enc2)
        dec3 = self.dec3(torch.cat((attn3, upconv3), dim=1))

        upconv2 = self.upconv2(dec3)
        attn2 = self.attn2(g=upconv2, x=enc1)
        dec2 = self.dec2(torch.cat((attn2, upconv2), dim=1))

        dec1 = self.dec1(torch.cat((attn2, upconv2), dim=1)) #Modifiye edildi

        # Final Convolution
        final_output = self.final_conv(dec1)

        return final_output

if __name__ == '__main__':
    # Modelin test edilmesi
    import torch
    # Giriş kanalı sayısı (örneğin, 1 kanal - gri tonlamalı)
    in_channels = 1
    # Sınıf sayısı (omurga, omurlar, omurlar arası diskler + arka plan)
    num_classes = 4
    # Bir örneğini oluştur
    model = AttentionUnet(in_channels=in_channels, num_classes=num_classes)

    # Rastgele bir giriş tensörü oluştur
    batch_size = 2
    time_steps = 5 # Zaman boyutu
    # Boyutlar dataset'teki resampling'e uygun olmalı
    input_tensor = torch.randn(batch_size, in_channels, time_steps, 192, 192)

    # Modeli çalıştır
    # output_tensor = model(input_tensor) #Model 3 boyutlu konvolüsyon kullandığı için aşağıdaki satırı kullanıyoruz
    output_tensor = model(input_tensor)

    # Boyutları kontrol et
    print("Giriş tensörü boyutu:", input_tensor.size())
    print("Çıkış tensörü boyutu:", output_tensor.size()) # [2, 4, 64, 64, 64] Olmalı