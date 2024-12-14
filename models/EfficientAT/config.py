from pydantic import BaseModel


class EfficientConfig(BaseModel):

    # AugmentMelSTFT
    n_mels: int = 256
    sr: int = 16000
    win_length: int = 400
    hopsize: int = 160
    n_fft: int = 512
    freqm: int = 20
    timem: int = 40
    fmin: int = 20
    fmax: int | None = 7000
    fmin_aug_range: int = 100
    fmax_aug_range: int = 4000

    # EnsemblerModel
    model_names: list[str] = ["mn10_as_mels_256"]

    # Model initialization
    ckpt_path: str | None = None
