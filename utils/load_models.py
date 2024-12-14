import torch

from src.entity import Checkpoint
from models import EfficientAT, EfficientConfig
from utils import from_dict_to_type


def load_efficientat(
    ckpt_path: str | None,
    device: torch.device,
) -> EfficientAT:

    model: EfficientAT

    if ckpt_path is not None:

        ckpt: Checkpoint
        try:
            ckpt = torch.load(
                f=ckpt_path,
                map_location=device,
                weights_only=False,
            )
        except RuntimeError:
            ckpt = torch.load(
                f=ckpt_path,
                weights_only=False,
            )

        model_cfgs: EfficientConfig = from_dict_to_type(
            dictionary=ckpt["hparams"],
        )
        model = EfficientAT(
            model_cfgs=model_cfgs,
        )

        model.load_state_dict(
            state_dict=ckpt["model"],
        )

    model.eval().to(device)

    return model
