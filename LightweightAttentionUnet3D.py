import torch
import torch.nn as nn

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__() #Depthwise + Pointwise Separable Convolution
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)#uzamsal bilgi çıkarılır. Kanallar arası bilgi yok
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)#kanallar arası bilgi de çıkarılır.
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class SEBlock3D(nn.Module):  #changed Attention Block, SEBlock (Squeeze-and-Excitation Attention)
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class LightweightAttentionUnet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        filters = [32, 64, 128, 256, 512]

        self.enc1 = nn.Sequential(DepthwiseSeparableConv3D(in_channels, filters[0]), SEBlock3D(filters[0]))
        self.enc2 = nn.Sequential(nn.MaxPool3d((1,2,2)), DepthwiseSeparableConv3D(filters[0], filters[1]), SEBlock3D(filters[1]))
        self.enc3 = nn.Sequential(nn.MaxPool3d((1,2,2)), DepthwiseSeparableConv3D(filters[1], filters[2]), SEBlock3D(filters[2]))
        self.enc4 = nn.Sequential(nn.MaxPool3d((1,2,2)), DepthwiseSeparableConv3D(filters[2], filters[3]), SEBlock3D(filters[3]))

        self.bottleneck = nn.Sequential(nn.MaxPool3d((1,2,2)), DepthwiseSeparableConv3D(filters[3], filters[4]), SEBlock3D(filters[4]))

        self.upconv4 = nn.ConvTranspose3d(filters[4], filters[3], (1,2,2), (1,2,2))
        self.dec4 = nn.Sequential(DepthwiseSeparableConv3D(filters[4], filters[3]), SEBlock3D(filters[3]))

        self.upconv3 = nn.ConvTranspose3d(filters[3], filters[2], (1,2,2), (1,2,2))
        self.dec3 = nn.Sequential(DepthwiseSeparableConv3D(filters[3], filters[2]), SEBlock3D(filters[2]))

        self.upconv2 = nn.ConvTranspose3d(filters[2], filters[1], (1,2,2), (1,2,2))
        self.dec2 = nn.Sequential(DepthwiseSeparableConv3D(filters[2], filters[1]), SEBlock3D(filters[1]))

        self.upconv1 = nn.ConvTranspose3d(filters[1], filters[0], (1,2,2), (1,2,2))
        self.dec1 = nn.Sequential(DepthwiseSeparableConv3D(filters[1], filters[0]), SEBlock3D(filters[0]))

        self.outconv = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)

        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))

        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.outconv(dec1)


if __name__ == '__main__':
    model = LightweightAttentionUnet3D(in_channels=1, out_channels=4)
    x = torch.randn(1, 1, 5, 192, 192)
    out = model(x)
    print("Output shape:", out.shape)
    print("Toplam parametre sayısı:", sum(p.numel() for p in model.parameters()))
