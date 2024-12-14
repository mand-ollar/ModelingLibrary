from typing import TypedDict, Mapping

import torch


class Checkpoint(TypedDict):

    model: Mapping[str, torch.Tensor]
    hparams: dict[str, int | str | list[str] | None]
