from __future__ import annotations

from logging import getLogger
from pathlib import Path

import PySimpleGUI as sg
import sounddevice as sd
import soundfile as sf
from pebble import ProcessPool

from .__main__ import init_logger

LOG = getLogger(__name__)

init_logger()


def play_audio(path: Path | str):
    if isinstance(path, Path):
        path = path.as_posix()
    data, sr = sf.read(path)
    sd.play(data, sr)


def main():
    sg.theme("Dark")
    model_candidates = list(sorted(Path("./logs/44k/").glob("G_*.pth")))
    layout = [
        [
            sg.Frame(
                "Paths",
                [
                    [
                        sg.Text("Model path"),
                        sg.Push(),
                        sg.InputText(
                            key="model_path",
                            default_text=model_candidates[-1].absolute().as_posix()
                            if model_candidates
                            else "",
                        ),
                        sg.FileBrowse(
                            initial_folder=Path("./logs/44k/").absolute
                            if Path("./logs/44k/").exists()
                            else Path(".").absolute().as_posix(),
                            key="model_path_browse",
                            file_types=(("PyTorch", "*.pth"),),
                        ),
                    ],
                    [
                        sg.Text("Config path"),
                        sg.Push(),
                        sg.InputText(
                            key="config_path",
                            default_text=Path("./configs/44k/config.json")
                            .absolute()
                            .as_posix()
                            if Path("./configs/44k/config.json").exists()
                            else "",
                            enable_events=True,
                        ),
                        sg.FileBrowse(
                            initial_folder=Path("./configs/44k/").as_posix()
                            if Path("./configs/44k/").exists()
                            else Path(".").absolute().as_posix(),
                            key="config_path_browse",
                            file_types=(("JSON", "*.json"),),
                        ),
                    ],
                    [
                        sg.Text("Cluster model path"),
                        sg.Push(),
                        sg.InputText(key="cluster_model_path"),
                        sg.FileBrowse(
                            initial_folder="./logs/44k/"
                            if Path("./logs/44k/").exists()
                            else ".",
                            key="cluster_model_path_browse",
                            file_types=(("PyTorch", "*.pth"),),
                        ),
                    ],
                ],
            )
        ],
        [
            sg.Frame(
                "Common",
                [
                    [
                        sg.Text("Speaker"),
                        sg.Combo(values=[], key="speaker", size=(20, 1)),
                    ],
                    [
                        sg.Text("Silence threshold"),
                        sg.Push(),
                        sg.Slider(
                            range=(-60.0, 0),
                            orientation="h",
                            key="silence_threshold",
                            default_value=-30,
                            resolution=0.1,
                        ),
                    ],
                    [
                        sg.Text("Pitch"),
                        sg.Push(),
                        sg.Slider(
                            range=(-20, 20),
                            orientation="h",
                            key="transpose",
                            default_value=0,
                        ),
                    ],
                    [
                        sg.Checkbox(
                            key="auto_predict_f0",
                            default=True,
                            text="Auto predict F0 (Pitch may become unstable when turned on in real-time inference.)",
                        )
                    ],
                    [
                        sg.Text("Cluster infer ratio"),
                        sg.Push(),
                        sg.Slider(
                            range=(0, 1.0),
                            orientation="h",
                            key="cluster_infer_ratio",
                            default_value=0,
                            resolution=0.01,
                        ),
                    ],
                    [
                        sg.Text("Noise scale"),
                        sg.Push(),
                        sg.Slider(
                            range=(0.0, 1.0),
                            orientation="h",
                            key="noise_scale",
                            default_value=0.4,
                            resolution=0.01,
                        ),
                    ],
                    [
                        sg.Text("Pad seconds"),
                        sg.Push(),
                        sg.Slider(
                            range=(0.0, 1.0),
                            orientation="h",
                            key="pad_seconds",
                            default_value=0.1,
                            resolution=0.01,
                        ),
                    ],
                    [
                        sg.Text("Chunk seconds"),
                        sg.Push(),
                        sg.Slider(
                            range=(0.0, 3.0),
                            orientation="h",
                            key="chunk_seconds",
                            default_value=0.5,
                            resolution=0.01,
                        ),
                    ],
                    [
                        sg.Checkbox(
                            key="absolute_thresh",
                            default=False,
                            text="Absolute threshold (ignored (True) in realtime inference)",
                        )
                    ],
                ],
            )
        ],
        [
            sg.Frame(
                "File",
                [
                    [
                        sg.Text("Input audio path"),
                        sg.Push(),
                        sg.InputText(key="input_path"),
                        sg.FileBrowse(initial_folder="."),
                        sg.Button("Play", key="play_input"),
                    ],
                    [sg.Checkbox(key="auto_play", default=True, text="Auto play")],
                ],
            )
        ],
        [
            sg.Frame(
                "Realtime",
                [
                    [
                        sg.Text("Crossfade seconds"),
                        sg.Push(),
                        sg.Slider(
                            range=(0, 0.6),
                            orientation="h",
                            key="crossfade_seconds",
                            default_value=0.1,
                            resolution=0.001,
                        ),
                    ],
                    [
                        sg.Text("Block seconds"),
                        sg.Push(),
                        sg.Slider(
                            range=(0, 3.0),
                            orientation="h",
                            key="block_seconds",
                            default_value=1,
                            resolution=0.01,
                        ),
                    ],
                    [
                        sg.Text("Realtime algorithm"),
                        sg.Combo(
                            ["2 (Divide by speech)", "1 (Divide constantly)"],
                            default_value="2 (Divide by speech)",
                            key="realtime_algorithm",
                        ),
                    ],
                ],
            )
        ],
        [sg.Checkbox(key="use_gpu", default=True, text="Use GPU")],
        [
            sg.Button("Infer", key="infer"),
            sg.Button("(Re)Start Voice Changer", key="start_vc"),
            sg.Button("Stop Voice Changer", key="stop_vc"),
        ],
    ]
    for row in layout:
        for frame in row:
            if isinstance(frame, sg.Frame):
                frame.expand_x = True
    window = sg.Window(
        f"{__name__.split('.')[0]}", layout
    )  # , use_custom_titlebar=True)
    with ProcessPool(max_workers=1) as pool:
        future = None
        while True:
            event, values = window.read(100)
            if event == sg.WIN_CLOSED:
                break

            def update_combo() -> None:
                from . import utils

                config_path = Path(values["config_path"])
                if config_path.exists() and config_path.is_file():
                    hp = utils.get_hparams_from_file(values["config_path"])
                    LOG.info(f"Loaded config from {values['config_path']}")
                    window["speaker"].update(
                        values=list(hp.__dict__["spk"].keys()), set_to_index=0
                    )

            if not event == sg.EVENT_TIMEOUT:
                LOG.info(f"Event {event}, values {values}")
            if values["speaker"] == "":
                update_combo()
            if event.endswith("_path"):
                browser = window[f"{event}_browse"]
                if isinstance(browser, sg.Button):
                    LOG.info(
                        f"Updating browser {browser} to {Path(values[event]).parent}"
                    )
                    browser.InitialFolder = Path(values[event]).parent
                    browser.update()
                else:
                    LOG.warning(f"Browser {browser} is not a FileBrowse")
            if event == "config_path":
                update_combo()
            elif event == "infer":
                from .inference_main import infer

                input_path = Path(values["input_path"])
                output_path = (
                    input_path.parent / f"{input_path.stem}.out{input_path.suffix}"
                )
                if not input_path.exists() or not input_path.is_file():
                    LOG.warning(f"Input path {input_path} does not exist.")
                    continue
                infer(
                    model_path=Path(values["model_path"]),
                    config_path=Path(values["config_path"]),
                    input_path=input_path,
                    output_path=output_path,
                    speaker=values["speaker"],
                    cluster_model_path=Path(values["cluster_model_path"])
                    if values["cluster_model_path"]
                    else None,
                    transpose=values["transpose"],
                    auto_predict_f0=values["auto_predict_f0"],
                    cluster_infer_ratio=values["cluster_infer_ratio"],
                    noise_scale=values["noise_scale"],
                    db_thresh=values["silence_threshold"],
                    pad_seconds=values["pad_seconds"],
                    absolute_thresh=values["absolute_thresh"],
                    chunk_seconds=values["chunk_seconds"],
                    device="cuda" if values["use_gpu"] else "cpu",
                )
                if values["auto_play"]:
                    pool.schedule(play_audio, args=[output_path])
            elif event == "play_input":
                if Path(values["input_path"]).exists():
                    pool.schedule(play_audio, args=[Path(values["input_path"])])
            elif event == "start_vc":
                from .inference_main import realtime

                if future:
                    LOG.info("Canceling previous task")
                    future.cancel()
                future = pool.schedule(
                    realtime,
                    kwargs=dict(
                        model_path=Path(values["model_path"]),
                        config_path=Path(values["config_path"]),
                        speaker=values["speaker"],
                        cluster_model_path=Path(values["cluster_model_path"])
                        if values["cluster_model_path"]
                        else None,
                        transpose=values["transpose"],
                        auto_predict_f0=values["auto_predict_f0"],
                        cluster_infer_ratio=values["cluster_infer_ratio"],
                        noise_scale=values["noise_scale"],
                        crossfade_seconds=values["crossfade_seconds"],
                        db_thresh=values["silence_threshold"],
                        pad_seconds=values["pad_seconds"],
                        chunk_seconds=values["chunk_seconds"],
                        version=int(values["realtime_algorithm"][0]),
                        device="cuda" if values["use_gpu"] else "cpu",
                        block_seconds=values["block_seconds"],
                    ),
                )
            elif event == "stop_vc":
                if future:
                    future.cancel()
                    future = None
        if future:
            future.cancel()
    window.close()
