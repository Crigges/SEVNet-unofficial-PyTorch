import torch
import torch.nn as nn
import math
from typing import Tuple

class ConstantScaling(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.eta = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / math.sqrt(self.channels)) * self.eta


class NormalizedSelfModulator(nn.Module):

    def __init__(self, channels: int, epsilon: float = 1e-6):
        super().__init__()
        self.affine_generator = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.affine_generator(x)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return x_normalized * gamma + beta


class HeterogeneousNormalizationModule(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        split_channels = channels // 2
        self.nsm_path = NormalizedSelfModulator(split_channels)
        self.cs_path = ConstantScaling(split_channels)
        self.channel_mixer = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2 = torch.chunk(x, 2, dim=1)
        out_nsm = self.nsm_path(f1)
        out_cs = self.cs_path(f2)
        concatenated = torch.cat([out_nsm, out_cs], dim=1)
        return self.channel_mixer(concatenated)


class InteractiveGatingModule(nn.Module):

    def __init__(self, channels: int, epsilon: float = 1e-6):
        super().__init__()
        split_channels = channels // 2
        self.value_conv = nn.Conv2d(split_channels, split_channels, kernel_size=3, padding=1)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2_gate = torch.chunk(x, 2, dim=1)
        f1_value = self.value_conv(f1)
        var_value = torch.var(f1_value, dim=1, keepdim=True, unbiased=False)
        var_gate = torch.var(f2_gate, dim=1, keepdim=True, unbiased=False)
        dss_term = torch.sqrt(var_value + var_gate + self.epsilon)
        return (f1_value * f2_gate) / dss_term


class ScaleEquivarianceBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.hnm = HeterogeneousNormalizationModule(channels)
        self.igm = InteractiveGatingModule(channels)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.hnm(x)
        x = self.igm(x)
        x = self.conv2(x)
        return x + residual


class DownsampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upconv(x)


class SEVNet(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 48,
                 num_blocks: Tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2)): # paper does not specify number of SEVs, using 2 as default
        super().__init__()
        ch = base_channels

        self.input_conv = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

        self.encoder1 = nn.Sequential(*[ScaleEquivarianceBlock(ch) for _ in range(num_blocks[0])])
        self.down1 = DownsampleBlock(ch, ch * 2)
        self.encoder2 = nn.Sequential(*[ScaleEquivarianceBlock(ch * 2) for _ in range(num_blocks[1])])
        self.down2 = DownsampleBlock(ch * 2, ch * 4)
        self.encoder3 = nn.Sequential(*[ScaleEquivarianceBlock(ch * 4) for _ in range(num_blocks[2])])
        self.down3 = DownsampleBlock(ch * 4, ch * 8)

        self.bottleneck = nn.Sequential(*[ScaleEquivarianceBlock(ch * 8) for _ in range(num_blocks[3])])

        self.up3 = UpsampleBlock(ch * 8, ch * 4)
        self.fusion3 = nn.Conv2d(ch * 8, ch * 4, kernel_size=1)
        self.decoder3 = nn.Sequential(*[ScaleEquivarianceBlock(ch * 4) for _ in range(num_blocks[4])])

        self.up2 = UpsampleBlock(ch * 4, ch * 2)
        self.fusion2 = nn.Conv2d(ch * 4, ch * 2, kernel_size=1)
        self.decoder2 = nn.Sequential(*[ScaleEquivarianceBlock(ch * 2) for _ in range(num_blocks[5])])

        self.up1 = UpsampleBlock(ch * 2, ch)
        self.fusion1 = nn.Conv2d(ch * 2, ch, kernel_size=1)
        self.decoder1 = nn.Sequential(*[ScaleEquivarianceBlock(ch) for _ in range(num_blocks[6])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_conv(x)
        s1 = self.encoder1(x1)

        x2 = self.down1(s1)
        s2 = self.encoder2(x2)

        x3 = self.down2(s2)
        s3 = self.encoder3(x3)

        x4 = self.down3(s3)

        b = self.bottleneck(x4)

        d3_up = self.up3(b)
        d3 = torch.cat([d3_up, s3], dim=1)
        d3 = self.fusion3(d3)
        d3 = self.decoder3(d3)

        d2_up = self.up2(d3)
        d2 = torch.cat([d2_up, s2], dim=1)
        d2 = self.fusion2(d2)
        d2 = self.decoder2(d2)

        d1_up = self.up1(d2)
        d1 = torch.cat([d1_up, s1], dim=1)
        d1 = self.fusion1(d1)
        d1 = self.decoder1(d1)

        out = self.output_conv(d1)
        return out

if __name__ == "__main__":
    model = SEVNet(in_channels=3, out_channels=3, base_channels=48)
    input_tensor = torch.randn(2, 3, 256, 256)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be [2, 3, 256, 256]
