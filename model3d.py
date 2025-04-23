import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttentionUnet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3): # out_channels=3 (omurlar, diskler, kanal)
        super(AttentionUnet3D, self).__init__()

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, 64)  # (1, 5, 192, 192) -> (64, 5, 192, 192)
        self.enc2 = nn.Sequential(nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)), DoubleConv3D(64, 128)) # (64, 5, 192, 192) -> (128, 5, 96, 96)
        self.enc3 = nn.Sequential(nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)), DoubleConv3D(128, 256)) # (128, 5, 96, 96) -> (256, 5, 48, 48)
        self.enc4 = nn.Sequential(nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)), DoubleConv3D(256, 512)) # (256, 5, 48, 48) -> (512, 5, 24, 24)

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)), DoubleConv3D(512, 1024)) # (512, 5, 24, 24) -> (1024, 5, 12, 12)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(1024, 512, kernel_size=(1,2,2), stride=(1,2,2)) # (1024, 5, 12, 12) -> (512, 5, 24, 24)
        self.attn4 = AttentionBlock3D(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv3D(1024, 512) # (1024, 5, 24, 24) -> (512, 5, 24, 24)

        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1,2,2), stride=(1,2,2)) # (512, 5, 24, 24) -> (256, 5, 48, 48)
        self.attn3 = AttentionBlock3D(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv3D(512, 256) # (512, 5, 48, 48) -> (256, 5, 48, 48)

        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=(1,2,2), stride=(1,2,2)) # (256, 5, 48, 48) -> (128, 5, 96, 96)
        self.attn2 = AttentionBlock3D(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv3D(256, 128) # (256, 5, 96, 96) -> (128, 5, 96, 96)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=(1,2,2), stride=(1,2,2)) # (128, 5, 96, 96) -> (64, 5, 192, 192)
        self.attn1 = AttentionBlock3D(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv3D(128, 64) # (128, 5, 192, 192) -> (64, 5, 192, 192)

        # Output
        self.outconv = nn.Conv3d(64, out_channels, kernel_size=1) # (64, 5, 192, 192) -> (3, 5, 192, 192)

        self.softmax = nn.Softmax(dim=1) # Sınıflandırma için Softmax

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        up4 = self.upconv4(bottleneck)
        attn4 = self.attn4(g=up4, x=enc4)
        dec4 = torch.cat((attn4, up4), dim=1)
        dec4 = self.dec4(dec4)

        up3 = self.upconv3(dec4)
        attn3 = self.attn3(g=up3, x=enc3)
        dec3 = torch.cat((attn3, up3), dim=1)
        dec3 = self.dec3(dec3)

        up2 = self.upconv2(dec3)
        attn2 = self.attn2(g=up2, x=enc2)
        dec2 = torch.cat((attn2, up2), dim=1)
        dec2 = self.dec2(dec2)

        up1 = self.upconv1(dec2)
        attn1 = self.attn1(g=up1, x=enc1)
        dec1 = torch.cat((attn1, up1), dim=1)
        dec1 = self.dec1(dec1)

        # Output
        output = self.outconv(dec1) # (batch_size, out_channels, depth, height, width)
        #output = output.squeeze(2) # Derinlik boyutunu kaldır (batch_size, out_channels, height, width)
        #output = self.softmax(output) # Sınıflandırma

        return output

if __name__ == '__main__':
    # Model Test
    model = AttentionUnet3D(in_channels=1, out_channels=3)
    input_tensor = torch.randn(1, 1, 5, 192, 192) # (batch_size, channels, depth, height, width)
    output = model(input_tensor)
    print("Output Shape:", output.shape)  #torch.Size([1, 3, 192, 192])

    # Parametre Sayısını Yazdırma
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")