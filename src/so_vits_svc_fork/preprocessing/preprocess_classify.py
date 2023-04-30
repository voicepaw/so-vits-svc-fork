from __future__ import annotations

from logging import getLogger
from pathlib import Path

import keyboard
import librosa
import sounddevice as sd
import soundfile as sf
from rich.console import Console
from tqdm.rich import tqdm

LOG = getLogger(__name__)


def preprocess_classify(
    input_dir: Path | str, output_dir: Path | str, create_new: bool = True
) -> None:
    # paths
    input_dir_ = Path(input_dir)
    output_dir_ = Path(output_dir)
    speed = 1
    if not input_dir_.is_dir():
        raise ValueError(f"{input_dir} is not a directory.")
    output_dir_.mkdir(exist_ok=True)

    console = Console()
    # get audio paths and folders
    audio_paths = list(input_dir_.glob("*.*"))
    last_folders = [x for x in output_dir_.glob("*") if x.is_dir()]
    console.print("Press ↑ or ↓ to change speed. Press any other key to classify.")
    console.print(f"Folders: {[x.name for x in last_folders]}")

    pbar_description = ""

    pbar = tqdm(audio_paths)
    for audio_path in pbar:
        # read file
        audio, sr = sf.read(audio_path)

        # update description
        duration = librosa.get_duration(y=audio, sr=sr)
        pbar_description = f"{duration:.1f} {pbar_description}"
        pbar.set_description(pbar_description)

        while True:
            # start playing
            sd.play(librosa.effects.time_stretch(audio, rate=speed), sr, loop=True)

            # wait for key press
            key = str(keyboard.read_key())
            if key == "down":
                speed /= 1.1
                console.print(f"Speed: {speed:.2f}")
            elif key == "up":
                speed *= 1.1
                console.print(f"Speed: {speed:.2f}")
            else:
                break

            # stop playing
            sd.stop()

        # print if folder changed
        folders = [x for x in output_dir_.glob("*") if x.is_dir()]
        if folders != last_folders:
            console.print(f"Folders updated: {[x.name for x in folders]}")
            last_folders = folders

        # get folder
        folder_candidates = [x for x in folders if x.name.startswith(key)]
        if len(folder_candidates) == 0:
            if create_new:
                folder = output_dir_ / key
            else:
                console.print(f"No folder starts with {key}.")
                continue
        else:
            if len(folder_candidates) > 1:
                LOG.warning(
                    f"Multiple folders ({[x.name for x in folder_candidates]}) start with {key}. "
                    f"Using first one ({folder_candidates[0].name})."
                )
            folder = folder_candidates[0]
        folder.mkdir(exist_ok=True)

        # move file
        new_path = folder / audio_path.name
        audio_path.rename(new_path)

        # update description
        pbar_description = f"Last: {audio_path.name} -> {folder.name}"

        # yield result
        # yield audio_path, key, folder, new_path
