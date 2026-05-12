import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up_conv = ConvBlock(in_channels, out_channels)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_conv(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class AutoTrashUNetDenoisingAE(nn.Module):
    """
    Compact U-Net style autoencoder for log-mel patches.

    Skip paths are deliberately narrowed with 1x1 convolutions before fusion.
    This gives the decoder local detail for stable reconstruction while avoiding a
    wide identity path that could bypass the latent bottleneck and reconstruct
    anomalies too easily.
    """

    def __init__(self, frames, n_mels, latent_dim=32, skip_scale=0.5):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels
        self.input_dim = frames * n_mels
        self.latent_dim = latent_dim
        self.skip_scale = skip_scale

        self.enc1 = ConvBlock(1, 8)
        self.enc2 = ConvBlock(8, 16)
        self.enc3 = ConvBlock(16, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.bottleneck = ConvBlock(32, 32)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, frames)
            b = self._encode_spatial(dummy)[-1]
        self.bottleneck_shape = tuple(b.shape[1:])
        bottleneck_dim = b.numel()

        self.fc_mu = nn.Linear(bottleneck_dim, latent_dim)
        self.fc_logvar = nn.Linear(bottleneck_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, bottleneck_dim)

        self.skip3 = nn.Conv2d(32, 8, kernel_size=1, bias=False)
        self.skip2 = nn.Conv2d(16, 8, kernel_size=1, bias=False)
        self.skip1 = nn.Conv2d(8, 4, kernel_size=1, bias=False)

        self.up3 = UpBlock(32, 8, 16)
        self.up2 = UpBlock(16, 8, 8)
        self.up1 = UpBlock(8, 4, 8)
        self.output = nn.Conv2d(8, 1, kernel_size=1)

    def _to_image(self, x):
        return x.view(-1, self.frames, self.n_mels).transpose(1, 2).unsqueeze(1)

    def _to_vector(self, x):
        return x.squeeze(1).transpose(1, 2).contiguous().view(-1, self.input_dim)

    def _encode_spatial(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self.bottleneck(self.pool(s3))
        return s1, s2, s3, b

    def encode(self, x):
        x = self._to_image(x)
        s1, s2, s3, b = self._encode_spatial(x)
        flat = b.flatten(start_dim=1)
        mu = self.fc_mu(flat)
        logvar = torch.clamp(self.fc_logvar(flat), min=-8.0, max=8.0)
        return s1, s2, s3, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z, s1, s2, s3):
        b = self.fc_decode(z).view(-1, *self.bottleneck_shape)
        s3 = self.skip_scale * self.skip3(s3)
        s2 = self.skip_scale * self.skip2(s2)
        s1 = self.skip_scale * self.skip1(s1)
        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.output(x)
        if x.shape[-2:] != (self.n_mels, self.frames):
            x = F.interpolate(x, size=(self.n_mels, self.frames), mode="bilinear", align_corners=False)
        return self._to_vector(x)

    def forward(self, x):
        s1, s2, s3, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, s1, s2, s3)
        return recon, z, mu, logvar
