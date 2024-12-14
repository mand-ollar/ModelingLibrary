import torch
import torch.nn as nn

from .config import EfficientConfig  # type: ignore
from .models.ensemble import EnsemblerModel, get_ensemble_model  # type: ignore
from .models.preprocess import AugmentMelSTFT  # type: ignore


class EfficientAT(nn.Module):

    def __init__(
        self,
        model_cfgs: EfficientConfig = EfficientConfig(),
    ) -> None:

        super().__init__()

        self.mel: AugmentMelSTFT = AugmentMelSTFT(
            n_mels=model_cfgs.n_mels,
            sr=model_cfgs.sr,
            win_length=model_cfgs.win_length,
            hopsize=model_cfgs.hopsize,
            n_fft=model_cfgs.n_fft,
            freqm=model_cfgs.freqm,
            timem=model_cfgs.timem,
            fmin=model_cfgs.fmin,
            fmax=model_cfgs.fmax,
            fmin_aug_range=model_cfgs.fmin_aug_range,
            fmax_aug_range=model_cfgs.fmax_aug_range,
        )
        self.model: EnsemblerModel = get_ensemble_model(
            model_names=model_cfgs.model_names,
        )

        self.device: torch.device

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.mel(x)
        output: tuple[torch.Tensor, torch.Tensor] = self.model(x)

        return output
