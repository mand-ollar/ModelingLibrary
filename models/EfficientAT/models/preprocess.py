import warnings

import torch
import torch.nn as nn
import torchaudio  # type: ignore
from torchaudio.transforms import FrequencyMasking, TimeMasking  # type: ignore


class AugmentMelSTFT(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 32000,
        win_length: int = 800,
        hopsize: int = 320,
        n_fft: int = 1024,
        freqm: int = 48,
        timem: int = 192,
        fmin: int = 0,
        fmax: int | None = None,
        fmin_aug_range: int = 10,
        fmax_aug_range: int = 2000,
    ) -> None:

        super().__init__()
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e

        self.n_mels: int = n_mels
        self.sr: int = sr
        self.win_length: int = win_length
        self.hopsize = hopsize
        self.n_fft: int = n_fft
        self.fmin: int = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            warnings.warn(f"Warning: FMAX is None setting to {fmax}")
        self.fmax: int = fmax

        self.register_buffer(
            name="window",
            tensor=torch.hann_window(win_length, periodic=False),
            persistent=False,
        )

        assert (
            fmin_aug_range >= 1
        ), f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert (
            fmax_aug_range >= 1
        ), f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"

        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer(
            name="preemphasis_coefficient",
            tensor=torch.as_tensor([[[-0.97, 1]]]),
            persistent=False,
        )

        self.freqm: nn.Identity | FrequencyMasking
        self.timem: nn.Identity | TimeMasking
        if freqm == 0:
            self.freqm = nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = nn.functional.conv1d(
            x.unsqueeze(1),
            self.preemphasis_coefficient,
        ).squeeze(1)
        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=False,
        )
        x = torch.view_as_real(x)
        x = (x**2).sum(dim=-1)  # power mag

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = (
            self.fmax
            + self.fmax_aug_range // 2
            - torch.randint(self.fmax_aug_range, (1,)).item()
        )

        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sr,
            fmin,
            fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0),
            device=x.device,
        )

        melspec: torch.Tensor
        melspec = torch.matmul(mel_basis, x)
        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec
