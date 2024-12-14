"""
Windowing Utility
=================

This module contains the windowing utility for the audio files.
From `.wav` & `.tsv` files, it creates windowed `.wav` files with labels & ratios information.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio  # type: ignore
from pydantic import BaseModel
from rich.progress import track
from torch.utils.data import Dataset
from tqdm import tqdm


# Windowing configuration
class WindowingConfig(BaseModel):
    audio_folders: str | Path | list[str | Path]
    similar_labels: dict[str, list[str]]
    window_size: float
    hop_size: float
    start_offset: float
    relative_ratio_threshold: float
    absolute_ratio_threshold: float
    classes: list[str]
    others: str | None


# Entity
class Window(BaseModel):
    window_st: int
    window_en: int
    iv_name: list[str]
    label_name: list[str]
    relative_ratio: list[float]
    absolute_ratio: list[float]


class WindowingResult(BaseModel):

    wav_path: str | Path
    audio: torch.Tensor
    windows: list[Window]

    class Config:
        arbitrary_types_allowed = True


# Windowed Dataset Class
class WindowedDataset(Dataset):

    sr: int = 16000

    def __init__(
        self,
        windowing_results: list[WindowingResult],
        window_sec: float,
        classes: list[str],
        others: str | None = None,
    ) -> None:
        """
        Args:
        -----
        windowing_results: list[WindowingResult]
            List of WindowingResult objects.
        classes: list[str]
            List of classes.
        others: str | None
            Label name for the others. If others is not specified, None.
        """

        self.windows: list[Window] = []

        self.windowing_results: list[WindowingResult] = windowing_results
        self.classes: list[str] = classes
        self.others: str | None = others

        self.audio_list: list[torch.Tensor] = []
        self.wav_path_list: list[Path] = []
        self.label_list: list[list[str]] = []

        for windowing_result in tqdm(windowing_results, ncols=80, leave=False):

            audio: torch.Tensor = windowing_result.audio
            windows: list[Window] = windowing_result.windows

            for window in windows:

                self.windows.append(window)

                windowed_audio: torch.Tensor = audio[
                    window.window_st : window.window_en
                ]
                margin: int = int(window_sec * self.sr - windowed_audio.size(0))

                windowed_audio = F.pad(
                    input=windowed_audio,
                    pad=(margin // 2, margin - margin // 2),
                    mode="constant",
                    value=0,
                )

                self.audio_list.append(windowed_audio)
                self.wav_path_list.append(Path(windowing_result.wav_path))
                self.label_list.append(window.iv_name)

    def __len__(
        self,
    ) -> int:
        return len(self.wav_path_list)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        audio: torch.Tensor = self.audio_list[idx].unsqueeze(dim=0)
        label: torch.Tensor = torch.zeros(len(self.classes), dtype=torch.long)
        for iv_name in self.label_list[idx]:
            if self.others is not None:
                label[self.classes.index(iv_name)] = 1
            else:
                if iv_name in self.classes:
                    label[self.classes.index(iv_name)] = 1

        return audio, label


# Windowing Class
class Windowing:

    sr: int = 16000

    def __init__(
        self,
        config: WindowingConfig,
    ) -> None:

        self.audio_folders: list[Path]
        if isinstance(config.audio_folders, (str, Path)):
            self.audio_folders = [Path(config.audio_folders)]
        else:
            self.audio_folders = [Path(folder) for folder in config.audio_folders]

        self.window_sec: float = config.window_size
        self.similar_labels: dict[str, list[str]] = config.similar_labels
        self.window_size: int = int(self.sr * config.window_size)
        self.hop_size: int = int(self.sr * config.hop_size)
        self.start_offset: int = int(self.sr * config.start_offset)
        self.relative_ratio_threshold: float = config.relative_ratio_threshold
        self.absolute_ratio_threshold: float = config.absolute_ratio_threshold
        self.classes: list[str] = config.classes
        self.others: str | None = config.others

        self.oov_list: list[str] = []

    def __gather_audio_files(
        self,
    ) -> list[Path]:
        """Gather audio files in the folder."""

        audio_files: list[Path] = []
        for audio_folder in self.audio_folders:
            audio_files += list(audio_folder.rglob("*.wav"))

        return audio_files

    def __short_mono_strong_label(
        self,
        audio_length: int,
        labels: list[str],
    ) -> Window | None:

        st, en, label_name = labels[0].split("\t")
        st_int = int(st)
        en_int = int(en)
        assert (
            st_int < en_int
        ), f"Start time must be smaller than end time: \n\t - Line 1, {label_name}.\n"

        # If strong label is shorter than the window size
        if en_int - st_int <= self.window_size:
            found = False
            for iv_name, similars in self.similar_labels.items():
                if label_name in similars:
                    iv_label_name = iv_name
                    found = True
                    break

            if not found:
                if label_name not in self.oov_list and label_name not in self.classes:
                    print(f" -- Considering\n{label_name}\nas others.\n")
                    self.oov_list.append(label_name)

                return None

            else:
                window: Window = Window(
                    window_st=max(0, (en_int + st_int) // 2 - self.window_size // 2),
                    window_en=min(
                        audio_length,
                        (en_int + st_int) // 2
                        - self.window_size // 2
                        + self.window_size,
                    ),
                    iv_name=[iv_label_name],
                    label_name=[label_name],
                    relative_ratio=[1.0],
                    absolute_ratio=[1.0],
                )

                return window

        else:
            return None

    def __otherwise(
        self,
        labels: list[str],
        window_st: int,
        window_en: int,
    ) -> Window | None:

        iv_name: list[str] = []
        label_name: list[str] = []
        relative_ratio: list[float] = []
        absolute_ratio: list[float] = []

        for i, label in enumerate(labels):
            st, en, label_name_i = label.split("\t")
            st_int: int = int(st)
            en_int: int = int(en)

            assert (
                st_int < en_int
            ), f"Start time must be smaller than end time: \n\t - Line {i+1}, {label}.\n"

            # When the strong label is overlapped with the window
            if st_int < window_en and en_int > window_st:

                # Calculate ratios
                overlap: float = min(en_int, window_en) - max(st_int, window_st)
                relative_ratio_i: float = overlap / self.window_size
                absolute_ratio_i: float = overlap / (en_int - st_int)

                found = False
                for iv_name_i, similars in self.similar_labels.items():

                    if label_name_i in similars and (
                        relative_ratio_i > self.relative_ratio_threshold
                        or absolute_ratio_i > self.absolute_ratio_threshold
                    ):
                        iv_name.append(iv_name_i)
                        label_name.append(label_name_i)
                        relative_ratio.append(relative_ratio_i)
                        absolute_ratio.append(absolute_ratio_i)

                        found = True
                        break

                # Out-of-vocabulary
                if not found:
                    if (
                        label_name_i not in self.oov_list
                        and label_name_i not in self.classes
                    ):
                        print(f" -- Considering\n{label_name_i}\nas others.\n")
                        self.oov_list.append(label_name_i)

            # When nothing is overlapped with the window
            else:
                pass

        if len(iv_name) == 0:
            return None

        else:
            window: Window = Window(
                window_st=window_st,
                window_en=window_en,
                iv_name=iv_name,
                label_name=label_name,
                relative_ratio=relative_ratio,
                absolute_ratio=absolute_ratio,
            )

            return window

    def __windowing(
        self,
        audio_filepath: str | Path,
        label_filepath: str | Path,
    ) -> WindowingResult:
        """
        Window the long audio file. Result is WindowingResult object.

        Arguments:
        ----------
        audio_filepath: str | Path
            Path to the audio file.
        label_filepath: str | Path
            Path to the label file.
            In sample point unit.
        """

        # Load audio and strong label
        audio, sr = torchaudio.load(uri=audio_filepath)
        audio = audio.squeeze()

        assert audio.dim() == 1, "Audio must be mono."
        assert sr == self.sr, f"Sample rate must be {self.sr}."

        with open(file=label_filepath, mode="r", encoding="utf-8") as f:
            labels = f.read().strip().split("\n")
            f.close()

        # Audio information for windowing
        audio_length: int = audio.shape[0]

        start_offset: int = int(self.start_offset * sr)
        assert (
            start_offset < self.window_size
        ), f"Start offset {start_offset} must be smaller than window size {self.window_size}."

        # Calculate the number of windows
        total_length: int = max(audio_length, self.window_size)
        num_windows: int = (total_length - self.hop_size) // self.hop_size + 1

        # Variables for windowing
        window: Window | None
        windows: list[Window] = []
        windowing_result: WindowingResult

        """ FIRST CASE
        Conditions:
        1. When the audio has only 1 strong label and the audio is longer than the window size
        2. However the length of strong label is shorter than or equal to the window size """

        if len(labels) == 1 and audio_length > self.window_size:
            window = self.__short_mono_strong_label(
                audio_length=audio_length,
                labels=labels,
            )

            # If Condition 2 is not satisfied, window is None and continue to the next part
            if isinstance(window, Window):
                windows.append(window)

                windowing_result = WindowingResult(
                    wav_path=audio_filepath,
                    audio=audio,
                    windows=windows,
                )

                return windowing_result

        """ THE OTHER CASES
        Conditions:
        1. When the audio is shorter than the window size
        2. When the audio has multiple labels
        3. When the audio is longer than the window size """

        # Pre-process the audio for windowing
        # Condition1 -> padding
        if audio_length < self.window_size:
            margin = self.window_size - audio_length
            audio = F.pad(
                input=audio,
                pad=(margin // 2, margin - margin // 2),
            )
            audio_length = audio.shape[0]

        # When start_offset != 0, num_windows can be changed.
        # Padding the audio and matching the num_windows are needed when you want to change start_offset and compare the results.
        if start_offset > 0:
            audio = F.pad(
                input=audio,
                pad=(0, start_offset),
            )

        for i in range(num_windows):
            window_st: int = max(0, start_offset + i * self.hop_size)
            window_en: int = min(audio_length, window_st + self.window_size)

            window = self.__otherwise(
                labels=labels,
                window_st=window_st,
                window_en=window_en,
            )
            if isinstance(window, Window):
                windows.append(window)

        windowing_result = WindowingResult(
            wav_path=audio_filepath,
            audio=audio,
            windows=windows,
        )

        return windowing_result

    def __call__(
        self,
    ) -> WindowedDataset:

        windowing_results: list[WindowingResult] = []

        audio_files: list[Path] = self.__gather_audio_files()

        for audio_file in track(audio_files):

            if audio_file.with_suffix(".tsv").exists():
                label_file = audio_file.with_suffix(".tsv")

            elif audio_file.with_suffix(".txt").exists():
                label_file = audio_file.with_suffix(".txt")

            windowing_result: WindowingResult = self.__windowing(
                audio_filepath=audio_file,
                label_filepath=label_file,
            )
            windowing_results.append(windowing_result)

        windowed_dataset: WindowedDataset = WindowedDataset(
            windowing_results=windowing_results,
            window_sec=self.window_sec,
            classes=self.classes,
            others=self.others,
        )

        return windowed_dataset
