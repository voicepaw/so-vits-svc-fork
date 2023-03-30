# 使用教程

## 推理

### 图形化界面

![GUI](https://raw.githubusercontent.com/34j/so-vits-svc-fork/main/docs/_static/gui.png)

请使用以下命令运行图形化界面:

```shell
svcg
```

### 命令行界面

- 实时转换 (输入源为麦克风)

```shell
svc vc --model-path <model-path>
```

- 从文件转换

```shell
svc --model-path <model-path> source.wav
```

[预训练模型](https://huggingface.co/models?search=so-vits-svc-4.0) 可以在 HuggingFace 获得。

### 注意

- 如果使用 WSL, 请注意 WSL 需要额外设置来处理音频，如果 GUI 找不到音频设备将不能正常工作。
- 在实时语音转换中, 如果输入源有杂音, HuBERT 模型依然会把杂音进行推理.可以考虑使用实时噪音减弱程序比如 [RTX Voice](https://www.nvidia.com/en-us/geforce/guides/nvidia-rtx-voice-setup-guide/)来解决.

## 训练

### 预处理

- 如果数据集有 BGM,请用例如[Ultimate Vocal Remover](https://ultimatevocalremover.com/)等软件去除 BGM.
  推荐使用`3_HP-Vocal-UVR.pth` 或者 `UVR-MDX-NET Main` . [^1]
- 如果数据集是包含多个歌手的长音频文件, 使用 `svc sd` 将数据集拆分为多个文件 (使用 `pyannote.audio`)
  。为了提高准确率，可能需要手动进行分类。如果歌手的声线多样,请把 --min-speakers 设置为大于实际说话者数量. 如果出现依赖未安装,
  请通过 `pip install pyannote-audio`来安装 `pyannote.audio`。
- 如果数据集是包含单个歌手的长音频文件, 使用 `svc split` 将数据集拆分为多个文件 (使用 `librosa`).

[^1]: https://ytpmv.info/how-to-use-uvr/

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/34j/so-vits-svc-fork/blob/main/notebooks/so-vits-svc-fork-4.0.ipynb)

### 本地

将数据集处理成 `dataset_raw/{speaker_id}/**/{wav_file}.{any_format}` 的格式(可以使用子文件夹和非 ASCII 文件名)然后运行:

```shell
svc pre-resample
svc pre-config
svc pre-hubert
svc train -t
```

### 注意

- 数据集的每个文件应该小于 10s，不然显存会爆。
- 如果想要 f0 的推理方式为 CREPE, 用 `svc pre-hubert -fm crepe` 替换 `svc pre-hubert`.由于性能原因，可能需要减少 `--n-jobs` 。
- 建议在执行 `train` 命令之前更改 `config.json` 中的 batch_size 以匹配显存容量。 默认值针对 Tesla T4（16GB 显存）进行了优化，但没有那么多显存也可以进行训练。
- 在原始仓库中，会自动移除静音和进行音量平衡，且这个操作并不是必须要处理的。

## 帮助菜单

更多命令, 运行 `svc -h` 或者 `svc <subcommand> -h`

```shell
> svc -h
用法: svc [OPTIONS] COMMAND [ARGS]...
  so-vits-svc 允许任何文件夹结构用于训练数据
  但是, 建议使用以下文件夹结构
      训练: dataset_raw/{speaker_name}/**/{wav_name}.{any_format}
      推理: configs/44k/config.json, logs/44k/G_XXXX.pth
  如果遵循文件夹结构,则无需指定模型路径,配置路径等,将自动加载最新模型
  若要要训练模型, 运行 pre-resample, pre-config, pre-hubert, train.
  若要要推理模型, 运行 infer.
可选:
  -h, --help  显示信息并退出
命令:
  clean          清理文件,仅在使用默认文件结构时有用
  infer          推理
  onnx           导出模型到onnx
  pre-config     预处理第 2 部分: config
  pre-hubert     预处理第 3 部分: 如果没有找到 HuBERT 模型,则会...
  pre-resample   预处理第 1 部分: resample
  pre-sd         Speech diarization 使用 pyannote.audio
  pre-split      将音频文件拆分为多个文件
  train          训练模型 如果 D_0.pth 或 G_0.pth 没有找到,自动从集线器下载.
  train-cluster  训练 k-means 聚类模型
  vc             麦克风实时推理
```

### 补充链接

[Youtube 视频教程](https://www.youtube.com/watch?v=tZn0lcGO5OQ)
