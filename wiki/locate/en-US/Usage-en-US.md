# Usage

## Inference

### GUI

![GUI](https://raw.githubusercontent.com/34j/so-vits-svc-fork/main/docs/_static/gui.png)

GUI launches with the following command:

```shell
svcg
```

### CLI

- Realtime (from microphone)

```shell
svc vc --model-path <model-path>
```

- File

```shell
svc --model-path <model-path> source.wav
```

[Pretrained models](https://huggingface.co/models?search=so-vits-svc-4.0) are available on HuggingFace.

### Notes

- If using WSL, please note that WSL requires additional setup to handle audio and the GUI will not work without finding an audio device.
- In real-time inference, if there is noise on the inputs, the HuBERT model will react to those as well. Consider using realtime noise reduction applications such as [RTX Voice](https://www.nvidia.com/en-us/geforce/guides/nvidia-rtx-voice-setup-guide/) in this case.

## Training

### Before training

- If your dataset has BGM, please remove the BGM using software such as [Ultimate Vocal Remover](https://ultimatevocalremover.com/). `3_HP-Vocal-UVR.pth` or `UVR-MDX-NET Main` is recommended. [^1]
- If your dataset is a long audio file with multiple speakers, use `svc sd` to split the dataset into multiple files (using `pyannote.audio`). Further manual classification may be necessary due to accuracy issues. If speakers speak with a variety of speech styles, set --min-speakers larger than the actual number of speakers. Due to unresolved dependencies, please install `pyannote.audio` manually: `pip install pyannote-audio`.
- If your dataset is a long audio file with a single speaker, use `svc split` to split the dataset into multiple files (using `librosa`).

[^1]: https://ytpmv.info/how-to-use-uvr/

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/34j/so-vits-svc-fork/blob/main/notebooks/so-vits-svc-fork-4.0.ipynb)

### Local

Place your dataset like `dataset_raw/{speaker_id}/**/{wav_file}.{any_format}` (subfolders and non-ASCII filenames are acceptable) and run:

```shell
svc pre-resample
svc pre-config
svc pre-hubert
svc train -t
```

### Notes

- Dataset audio duration per file should be <~ 10s or VRAM will run out.
- To change the f0 inference method to CREPE, replace `svc pre-hubert` with `svc pre-hubert -fm crepe`. You may need to reduce `--n-jobs` due to performance issues.
- It is recommended to change the batch_size in `config.json` before the `train` command to match the VRAM capacity. The default value is optimized for Tesla T4 (16GB VRAM), but training is possible without that much VRAM.
- Silence removal and volume normalization are automatically performed (as in the upstream repo) and are not required.

## Further help

For more details, run `svc -h` or `svc <subcommand> -h`.

```shell
> svc -h
Usage: svc [OPTIONS] COMMAND [ARGS]...
  so-vits-svc allows any folder structure for training data.
  However, the following folder structure is recommended.
      When training: dataset_raw/{speaker_name}/**/{wav_name}.{any_format}
      When inference: configs/44k/config.json, logs/44k/G_XXXX.pth
  If the folder structure is followed, you DO NOT NEED TO SPECIFY model path, config path, etc.
  (The latest model will be automatically loaded.)
  To train a model, run pre-resample, pre-config, pre-hubert, train.
  To infer a model, run infer.
Options:
  -h, --help  Show this message and exit.
Commands:
  clean          Clean up files, only useful if you are using the default file structure
  infer          Inference
  onnx           Export model to onnx
  pre-config     Preprocessing part 2: config
  pre-hubert     Preprocessing part 3: hubert If the HuBERT model is not found, it will be...
  pre-resample   Preprocessing part 1: resample
  pre-sd         Speech diarization using pyannote.audio
  pre-split      Split audio files into multiple files
  train          Train model If D_0.pth or G_0.pth not found, automatically download from hub.
  train-cluster  Train k-means clustering
  vc             Realtime inference from microphone
```
### External Links
[Video Tutorial](https://www.youtube.com/watch?v=tZn0lcGO5OQ)