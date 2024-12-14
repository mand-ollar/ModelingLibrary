import torch
import torch.nn as nn

from ..helpers.utils import NAME_TO_WIDTH  # type: ignore
from .dymn.model import get_model as get_dymn  # type: ignore
from .mn.model import get_model as get_mobilenet  # type: ignore


class EnsemblerModel(nn.Module):
    def __init__(
        self,
        models: list[nn.Module],
    ) -> None:

        super().__init__()

        self.models: nn.ModuleList = nn.ModuleList(models)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        all_out: torch.Tensor | None = None

        for m in self.models:

            out: torch.Tensor
            out, _ = m(x)

            if all_out is None:
                all_out = out
            else:
                all_out = out + all_out

        assert all_out is not None

        all_out = all_out / len(self.models)

        return all_out, all_out


def get_ensemble_model(
    model_names: list[str],
) -> EnsemblerModel:

    models: list[nn.Module] = []

    model: nn.Module
    for model_name in model_names:
        if model_name.startswith("dymn"):
            model = get_dymn(
                width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name
            )
        else:
            model = get_mobilenet(
                width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name
            )
        models.append(model)

    return EnsemblerModel(models)
