from pathlib import Path

import h5py  # type: ignore
import numpy as np
from tqdm import tqdm

from utils import WindowedDataset, Windowing, WindowingConfig


H5_KEY_DTYPE: dict[str, h5py.Datatype] = {
    "wav": h5py.special_dtype(vlen=np.float32),
    "label": h5py.special_dtype(vlen=np.int64),
    "iv_name": h5py.string_dtype(),
    "label_name": h5py.string_dtype(),
}


def pack_to_h5(
    windowing_config: WindowingConfig,
    h5_filepath: str | Path,
) -> None:

    windowed_dataset: WindowedDataset = Windowing(
        config=windowing_config,
    )()

    dataset_length: int = len(windowed_dataset)

    with h5py.File(name=h5_filepath, mode="w") as h5_file:

        for key, dtype in H5_KEY_DTYPE.items():
            h5_file.create_dataset(
                name=key,
                shape=(dataset_length,),
                dtype=dtype,
                chunks=True,
                maxshape=(None,),
            )

        for i in tqdm(range(dataset_length), ncols=80):

            wav, label = windowed_dataset[i]

            h5_file["wav"][i] = wav
            h5_file["label"][i] = label
            h5_file["iv_name"][i] = windowed_dataset.windows[i].iv_name[0]
            h5_file["label_name"][i] = windowed_dataset.windows[i].label_name[0]
