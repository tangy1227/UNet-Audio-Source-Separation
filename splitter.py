import math
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# from unet import UNet, UNet2
from unet_copy import UNet
from data import musdbDataset

def batchify(tensor: Tensor, T: int) -> Tensor:
    """
    partition tensor into segments of length T, zero pad any ragged samples
    Args:
        tensor(Tensor): BxCxFxL
    Returns:
        tensor of size (B*[L/T] x C x F x T)
    """
    # Zero pad the original tensor to an even multiple of T
    orig_size = tensor.size(-1)
    new_size = math.ceil(orig_size / T) * T
    tensor = F.pad(tensor, [0, new_size - orig_size])
    # Partition the tensor into multiple samples of length T and stack them into a batch
    return torch.cat(torch.split(tensor, T, dim=-1), dim=0)

class Splitter(nn.Module):
    def __init__(self, stem_names: List[str] = None):
        super(Splitter, self).__init__()

        assert stem_names, "Must provide stem names."
        # stft config
        self.F = 1024
        self.T = 512
        self.win_length = 4096
        self.hop_length = 1024
        self.win = nn.Parameter(torch.hann_window(self.win_length), requires_grad=False).to("cuda")

        # Create Unet for each stem
        self.stems = nn.ModuleDict({name: UNet(in_channels=2) for name in stem_names}).to("cuda")

    def compute_stft(self, wav: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes stft feature from wav
        Args:
            wav (Tensor): B x L
        """
        stft = torch.stft(
            wav,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            window=self.win,
            center=True,
            return_complex=False,
            pad_mode="reflect",
        )

        # only keep freqs smaller than self.F
        stft = stft[:, 0:self.F, :, :]
        real = stft[:, :, :, 0]
        im = stft[:, :, :, 1]
        mag = torch.sqrt(real**2 + im**2)
        phase = torch.atan2(im, real)
        return stft, mag, phase

    def inverse_stft(self, stft: Tensor) -> Tensor:
        """Inverses stft to wave form"""

        pad = self.win_length // 2 + 1 - stft.size(1)   # 4096 // 2 + 1 - 1024
        # stft = F.pad(stft, (0, 0, 0, 0, 0, pad))        # For testing spectrogram
        stft = F.pad(stft, (0, 0, 0, pad))            # For testing magnitude spectrogram
        wav = torch.istft(
            stft,
            self.win_length,
            hop_length=self.hop_length,
            center=True,
            window=self.win,
        )
        return wav.detach()

    def forward(self, wav: Tensor) -> Dict[str, Tensor]:
        """
        Separates stereo wav into different tracks (1 predicted track per stem)
        Args:
            wav (tensor): 2 x L
        Returns:
            masked stfts by track name
        """
        # stft - 2 X F x L x 2
        # stft_mag - 2 X F x L
        stft, stft_mag, phase_mix = self.compute_stft(wav.squeeze())
        stft_mag_mix = stft_mag

        L = stft.size(2)

        # 1 x 2 x F x T
        stft_mag = stft_mag.unsqueeze(-1).permute([3, 0, 1, 2])
        stft_mag = batchify(stft_mag, self.T)  # B x 2 x F x T
        stft_mag = stft_mag.transpose(2, 3)  # B x 2 x T x F

        # compute stems' mask
        masks = {name: net(stft_mag) for name, net in self.stems.items()} # use stft_mag input to the network and estimate mask output

        # compute denominator
        mask_sum = sum([m**2 for m in masks.values()])
        mask_sum += 1e-10

        ############ uncomment for testing preprocessing.ipynb ############
        # return masks

        def apply_mask(mask):
            mask = (mask**2 + 1e-10 / 2) / (mask_sum)
            mask = mask.transpose(2, 3)  # B x 2 X F x T
            mask = torch.cat(torch.split(mask, 1, dim=0), dim=3)

            # mask = mask.squeeze(0)[:, :, :L].unsqueeze(-1)  # 2 x F x L x 1
            # stft_masked = stft * mask

            mask = mask.squeeze(0)[:, :, :L]
            src_magnitude = stft_mag_mix * mask
            stft_masked = src_magnitude * torch.exp(1j * phase_mix)
            return stft_masked

        return {name: apply_mask(m) for name, m in masks.items()}

    def separate(self, wav: Tensor) -> Dict[str, Tensor]:
        """
        Separates stereo wav into different tracks (1 predicted track per stem)
        Args:
            wav (tensor): 2 x L
        Returns:
            wavs by track name
        """

        stft_masks = self.forward(wav)

        return {
            name: self.inverse_stft(stft_masked)
            for name, stft_masked in stft_masks.items()
        }

if __name__ == "__main__":
    # train_dataset = musdbDataset(root="/Users/Owen/Desktop/MUSI7100_Spr2023/musdb18/musdb18")
    # for stem in train_dataset.targets:
    #     print(stem)

    # Create Each Instrument Model 
    model = Splitter(stem_names=["vocals"])

    train_dataset = musdbDataset()
    # train_dataset = MusdbDataset()
    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size = 1
    )
    batch_iterator = tqdm(train_dataloader, desc="Batch")

    for batch_idx, batch in enumerate(batch_iterator):
        # batch[0]: Tensor([Batch, Channels, Length]) --> x_wav
        # batch[1]: {"instrument": Tensor[batch_size, channels, sample_len]} --> y_target_wav
        x_wav, y_target_wav = batch
        if batch_idx == 0:
            print(x_wav.size())
            reduce_x_wav = x_wav.squeeze()
            print(reduce_x_wav.size())
            reduce2_x_wav = torch.cat((reduce_x_wav[0], reduce_x_wav[1]), dim=0)
            print(reduce2_x_wav.size())

            stft2 = torch.stft(
                x_wav.squeeze(),
                n_fft=2048, # 4096
                hop_length=1024,
                window=nn.Parameter(torch.hann_window(2048), requires_grad=False),
                center=True,
                return_complex=False,
                pad_mode="reflect",
            )
            print("stft size:",stft2.size())

            stft2 = stft2[:, 0:1024, :, :]
            real = stft2[:, :, :, 0]
            im = stft2[:, :, :, 1]
            mag = torch.sqrt(real**2 + im**2)
            phase = torch.atan2(im, real)
            print("mag size:", mag.size())
            print("phase size:", phase.size())

            stft_mag = mag.unsqueeze(-1).permute([3, 0, 1, 2])
            stft_mag = batchify(stft_mag, 512)  # B x 2 x F x T
            stft_mag = stft_mag.transpose(2, 3)  # B x 2 x T x F

            print(stft_mag.size())

        else:
            break