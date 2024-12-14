from src.make_dataset import pack_to_h5
from utils import WindowingConfig

from ..assets import similar_labels  # type: ignore

windowing_config: WindowingConfig = WindowingConfig(
    audio_folders=[
        "./data/anchor_v0.3",
    ],
    similar_labels=similar_labels,
    window_size=2.0,
    hop_size=1.0,
    start_offset=0.0,
    relative_ratio_threshold=0.3,
    absolute_ratio_threshold=0.9,
    classes=[
        "breathing_heavily",
        "crying_sobbing",
        "scream_female",
        "scream_male",
        "aggressive_yell_female",
        "aggressive_yell_male",
        "normal_speech_female",
        "normal_speech_male",
        "physical_impacts",
        "glass_break",
    ],
    others="others",
)

h5_filepath: str = "./data/anchor_v0.3/anchor_v0.3.h5"
