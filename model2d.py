import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
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

class Conv2DAttentionUnet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1): # in_channels=5
        super(Conv2DAttentionUnet, self).__init__()

        # Encoder
        self.enc1 = DoubleConv2D(in_channels, 64)
        self.enc2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv2D(64, 128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv2D(128, 256))
        self.enc4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv2D(256, 512))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv2D(512, 1024))

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attn4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv2D(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv2D(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv2D(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv2D(128, 64)

        # Output
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

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
        output = self.outconv(dec1)
        return output

if __name__ == '__main__':
    # Conv2D Modeli Test
    model_2d = Conv2DAttentionUnet(in_channels=5, out_channels=1)
    input_tensor = torch.randn(1, 5, 192, 192) # (batch_size, channels, height, width)
    output_2d = model_2d(input_tensor)
    print("Conv2D Output Shape:", output_2d.shape)

    # Parametre Sayısını Yazdırma
    total_params = sum(p.numel() for p in model_2d.parameters())
    print(f"Conv2D Total Parameters: {total_params}")