from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchinfo import summary 

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # in_c: input channel, out_c: outpu channel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            track_running_stats=True,
            eps=1e-3,
            momentum=0.01
        )
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        down = self.conv(F.pad((input), (1, 2, 1, 2), "constant", 0))
        return down, self.relu(self.bn(down))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.0) -> None:
        super().__init__()

        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            track_running_stats=True,
            eps=1e-3,
            momentum=0.01
        )
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = nn.Identity()
    
    def forward(self, input: Tensor) -> Tensor:
        up = self.tconv(input)
        l, r, t, b = 1, 2, 1, 2
        up = up[:, :, l:-r, t:-b]
        out_up = self.dropout(self.bn(self.relu(up)))
        return out_up

class CustomPad(nn.Module):
    def __init__(self, padding_setting=(1,2,1,2)) -> None:
        super(CustomPad, self).__init__()
        self.padding_setting = padding_setting
    def forward(self, x):
        return F.pad(x, self.padding_setting, "constant", 0)

class CustomTransposedPad(nn.Module):
    def __init__(self, padding_setting=(1,2,1,2)) -> None:
        super(CustomTransposedPad, self).__init__()
        self.padding_setting = padding_setting
    def forward(self, x):
        l, r, t, b = self.padding_setting
        return x[:, :, l:-r, t:-b]

def encoder_block(in_channels, out_channels):
    down_conv = nn.Sequential(
        CustomPad(),
        nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=0)
    )

    for m in down_conv:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    down_act = nn.Sequential(
        nn.BatchNorm2d(out_channels, track_running_stats=True, eps=1e-3, momentum=0.01),
        nn.LeakyReLU(0.2)
    )
    return down_conv, down_act

def decoder_block(in_c, out_c, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, kernel_size=5, stride=2),
        CustomTransposedPad(),
        nn.ReLU(),
        nn.BatchNorm2d(out_c, track_running_stats=True, eps=1e-3, momentum=0.01)
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    up = nn.Sequential(*layers)

    for m in up:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            
    return up

class UNet(nn.Module):
    def __init__(self, in_channels: int=2) -> None:
        super().__init__()

        self.down1_conv, self.down1_act = encoder_block(in_channels, out_channels=16)
        self.down2_conv, self.down2_act = encoder_block(in_channels=16, out_channels=32)
        self.down3_conv, self.down3_act = encoder_block(in_channels=32, out_channels=64)
        self.down4_conv, self.down4_act = encoder_block(in_channels=64, out_channels=128)
        self.down5_conv, self.down5_act = encoder_block(in_channels=128, out_channels=256)
        self.down6_conv, self.down6_act = encoder_block(in_channels=256, out_channels=512)

        self.up1 = decoder_block(512, 256, dropout=True)
        self.up2 = decoder_block(512, 128, dropout=True)
        self.up3 = decoder_block(256, 64, dropout=True)
        self.up4 = decoder_block(128, 32, dropout=False)
        self.up5 = decoder_block(64, 16, dropout=False)
        self.up6 = decoder_block(32, 1, dropout=False)
        self.up7 = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=4, dilation=2, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        d1_conv = self.down1_conv(x)
        d1 = self.down1_act(d1_conv)

        d2_conv = self.down2_conv(d1)
        d2 = self.down2_act(d2_conv)

        d3_conv = self.down3_conv(d2)
        d3 = self.down3_act(d3_conv)

        d4_conv = self.down4_conv(d3)
        d4 = self.down4_act(d4_conv)

        d5_conv = self.down5_conv(d4)
        d5 = self.down5_act(d5_conv)

        d6_conv = self.down6_conv(d5)
        d6 = self.down6_act(d6_conv)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat([d5_conv, u1], axis=1))
        u3 = self.up3(torch.cat([d4_conv, u2], axis=1))
        u4 = self.up4(torch.cat([d3_conv, u3], axis=1))
        u5 = self.up5(torch.cat([d2_conv, u4], axis=1))
        u6 = self.up6(torch.cat([d1_conv, u5], axis=1))
        u7 = self.up7(u6)
        return u7 * x

class UNet2(nn.Module):
    def __init__(
        self,
        n_layers: int = 8,
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        # DownSample layers
        # down_set = [in_c] + [2 ** (i + 4) for i in range(n_layers)] # [1, 16, 32, 64 ...]
        down_set = [in_channels] + [2 ** (i + 4) for i in range(n_layers)]
        self.encoder_layers = nn.ModuleList(
            [
                # in_c = in_channel, out_c = 16
                # in_c = 16, out_c = 32
                # in_c = 32, out_c = 64
                EncoderBlock(in_channels=in_ch, out_channels=out_ch)
                for in_ch, out_ch in zip(down_set[:-1], down_set[1:])
            ]
        )

        # UpSample layers
        up_set = [1] + [2 ** (i + 4) for i in range(n_layers)]
        up_set.reverse()
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    # doubled for concatenated inputs (skip connections)
                    in_channels=in_ch if i == 0 else in_ch * 2,
                    out_channels=out_ch,
                    #   50 % dropout... first 3 layers only
                    dropout_prob=0.5 if i < 3 else 0,
                )
                for i, (in_ch, out_ch) in enumerate(zip(up_set[:-1], up_set[1:]))
            ]
        )

        # reconstruct the final mask same as the original channels
        self.up_final = nn.Conv2d(1, in_channels, kernel_size=4, dilation=2, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        encoder_outputs_pre_act = []
        x = input
        for down in self.encoder_layers:
            conv, x = down(x)
            encoder_outputs_pre_act.append(conv)

        for i, up in enumerate(self.decoder_layers):
            if i == 0:
                x = up(encoder_outputs_pre_act.pop())
            else:
                # merge skip connection
                x = up(torch.concat([encoder_outputs_pre_act.pop(), x], axis=1))
        mask = self.sigmoid(self.up_final(x))
        return mask * input


if __name__ == "__main__":

    net = UNet()
    print(net)
    # summary(net)
