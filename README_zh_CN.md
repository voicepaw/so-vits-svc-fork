# SoftVC VITS Singing Voice Conversion

<p align="center">
  <a href="https://github.com/34j/so-vits-svc-fork/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/so-vits-svc-fork/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://so-vits-svc-fork.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/so-vits-svc-fork.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/so-vits-svc-fork">
    <img src="https://img.shields.io/codecov/c/github/34j/so-vits-svc-fork.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/so-vits-svc-fork/">
    <img src="https://img.shields.io/pypi/v/so-vits-svc-fork.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/so-vits-svc-fork.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/so-vits-svc-fork.svg?style=flat-square" alt="License">
</p>

基于 [`so-vits-svc4.0(V1)`](https://github.com/svc-develop-team/so-vits-svc)的一个分支，支持实时推理和图形化推理界面，且兼容其模型。

## 新功能

- **实时语音转换** (增强版本 v1.1.0)
- 与[`QuickVC`](https://github.com/quickvc/QuickVC-VoiceConversion)相结合
- 修复了原始版本中对 [`ContentVec`](https://github.com/auspicious3000/contentvec) 的误用[^c]
- 使用 CREPE 进行更准确的音高推测
- 图形化界面和统一命令行界面
- 相比之前双倍的训练速度
- 只需使用 `pip` 安装即可使用，不需要安装 `fairseq`
- 自动下载预训练模型和 HuBERT 模型
- 使用 black、isort、autoflake 等完全格式化的代码

[^c]: [#206](https://github.com/34j/so-vits-svc-fork/issues/206)

## 安装教程

### 可以使用 bat 一键安装

<a href="https://github.com/xieyumc/so-vits-svc-fork-cn/releases/download/install/install-cn.bat" download>
  <img src="https://img.shields.io/badge/.bat-download-blue?style=flat-square&logo=windows" alt="Download .bat">
</a>

### 本 bat 汉化基于英文版，对原版进行了一些本地工作和优化，如安装过程有问题，可以尝试安装原版

<a href="https://github.com/34j/so-vits-svc-fork/releases/download/v1.3.2/install.bat" download>
  <img src="https://img.shields.io/badge/.bat-download-blue?style=flat-square&logo=windows" alt="Download .bat">
</a>

### 手动安装

<details>
  <summary>创建一个虚拟环境</summary>

Windows:

```shell
py -3.10 -m venv venv
venv\Scripts\activate
```

Linux/MacOS:

```shell
python3.10 -m venv venv
source venv/bin/activate
```

Anaconda:

```shell
conda create -n so-vits-svc-fork python=3.10 pip
conda activate so-vits-svc-fork
```

如果 Python 安装在 Program Files，在安装时未创造虚拟环境可能会导致`PermissionError`

</details>

### 安装

通过 pip 安装 (或者通过包管理器使用 pip 安装):

```shell
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U so-vits-svc-fork
```

- 如果没有可用 GPU 或使用 MacOS, 不需要执行 `pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118`. MPS 可能已经安装了.
- 如果在 Linux 下使用 AMD GPU, 请使用此命令 `--index-url https://download.pytorch.org/whl/rocm5.4.2`
  替换掉 `--index-url https://download.pytorch.org/whl/cu118` . Windows 下不支持 AMD GPUs (#120).

### 更新

请经常更新以获取最新功能和修复错误:

```shell
pip install -U so-vits-svc-fork
```

## 使用教程

### 推理

#### 图形化界面

![GUI](https://raw.githubusercontent.com/34j/so-vits-svc-fork/main/docs/_static/gui.png)

请使用以下命令运行图形化界面:

```shell
svcg
```

#### 命令行界面

- 实时转换 (输入源为麦克风)

```shell
svc vc
```

- 从文件转换

```shell
svc infer source.wav
```

[预训练模型](https://huggingface.co/models?search=so-vits-svc-4.0) 可以在 HuggingFace 获得。

#### 注意

- 如果使用 WSL, 请注意 WSL 需要额外设置来处理音频，如果 GUI 找不到音频设备将不能正常工作。
- 在实时语音转换中, 如果输入源有杂音, HuBERT
  模型依然会把杂音进行推理.可以考虑使用实时噪音减弱程序比如 [RTX Voice](https://www.nvidia.com/en-us/geforce/guides/nvidia-rtx-voice-setup-guide/)
  来解决.

### 训练

#### 预处理

- 如果数据集有 BGM,请用例如[Ultimate Vocal Remover](https://ultimatevocalremover.com/)等软件去除 BGM.
  推荐使用`3_HP-Vocal-UVR.pth` 或者 `UVR-MDX-NET Main` . [^1]
- 如果数据集是包含单个歌手的长音频文件, 使用 `svc pre-split` 将数据集拆分为多个文件 (使用 `librosa`).
- 如果数据集是包含多个歌手的长音频文件, 使用 `svc pre-sd` 将数据集拆分为多个文件 (使用 `pyannote.audio`)
  。为了提高准确率，可能需要手动进行分类。如果歌手的声线多样,请把 --min-speakers 设置为大于实际说话者数量. 如果出现依赖未安装,
  请通过 `pip install pyannote-audio`来安装 `pyannote.audio`。

[^1]: https://ytpmv.info/how-to-use-uvr/

#### 云端

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/34j/so-vits-svc-fork/blob/main/notebooks/so-vits-svc-fork-4.0.ipynb)
[![Open In Paperspace](https://img.shields.io/badge/Open%20in-Paperspace-blue?style=flat-square&logo=paperspace)](https://console.paperspace.com/github/34j/so-vits-svc-fork-paperspace/blob/main/so-vits-svc-fork-4.0-paperspace.ipynb)
[![Paperspace Referral](<https://img.shields.io/badge/Referral%20($10)-9VJN74I-blue?style=flat-square&logo=paperspace>)](https://www.paperspace.com/?r=9VJN74I)[^p]

如果你无法获取 10GB 显存以上的显卡，对于轻量用户，推荐使用 Google Colab 的免费方案；而重度用户，则建议使用 Paperspace 的 Pro/Growth Plan。当然，如果你有高端的显卡，就没必要使用云服务了。

[^p]: If you register a referral code and then add a payment method, you may save about $5 on your first month's monthly billing. Note that both referral rewards are Paperspace credits and not cash. It was a tough decision but inserted because debugging and training the initial model requires a large amount of computing power and the developer is a student.

#### 本地

将数据集处理成 `dataset_raw/{speaker_id}/**/{wav_file}.{any_format}` 的格式(可以使用子文件夹和非 ASCII 文件名)然后运行:

```shell
svc pre-resample
svc pre-config
svc pre-hubert
svc train -t
```

#### 注意

- 数据集的每个文件应该小于 10s，不然显存会爆。
- 建议在执行 `train` 命令之前提高 `config.json` 中的 `batch_size` 以匹配显存容量。 将`batch_size`设为`auto-{init_batch_size}-{max_n_trials}`（或者只需设为`auto`）就会自动提高`batch_size`，直到爆显存为止（不过自动调高 batch_size 有概率失效）
- 如果想要 f0 的推理方式为 `CREPE`, 用 `svc pre-hubert -fm crepe` 替换 `svc pre-hubert`.
- 若想正确使用`ContentVec`，用 `-t so-vits-svc-4.0v1`替换`svc pre-config`。由于复用 generator weights，一些 weights 会被重置而导致训练时间稍微延长.
- 若要使用`MS-iSTFT Decoder`，用 `svc pre-config -t quickvc`替换 `svc pre-config`.
- 在原始仓库中，会自动移除静音和进行音量平衡，且这个操作并不是必须要处理的。
- 倘若你已经大规模训练了一个免费公开版权的数据集，可以考虑将其作为底模发布。
- 对于更多细节（比如参数等），详见[Wiki](https://github.com/34j/so-vits-svc-fork/wiki) 或 [Discussions](https://github.com/34j/so-vits-svc-fork/discussions).

### 帮助

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

#### 补充链接

[视频教程](https://www.youtube.com/watch?v=tZn0lcGO5OQ)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/34j"><img src="https://avatars.githubusercontent.com/u/55338215?v=4?s=80" width="80px;" alt="34j"/><br /><sub><b>34j</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=34j" title="Code">💻</a> <a href="#ideas-34j" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=34j" title="Documentation">📖</a> <a href="#example-34j" title="Examples">💡</a> <a href="#infra-34j" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-34j" title="Maintenance">🚧</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/pulls?q=is%3Apr+reviewed-by%3A34j" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=34j" title="Tests">⚠️</a> <a href="#tutorial-34j" title="Tutorials">✅</a> <a href="#promotion-34j" title="Promotion">📣</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3A34j" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GarrettConway"><img src="https://avatars.githubusercontent.com/u/22782004?v=4?s=80" width="80px;" alt="GarrettConway"/><br /><sub><b>GarrettConway</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=GarrettConway" title="Code">💻</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3AGarrettConway" title="Bug reports">🐛</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=GarrettConway" title="Documentation">📖</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/pulls?q=is%3Apr+reviewed-by%3AGarrettConway" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BlueAmulet"><img src="https://avatars.githubusercontent.com/u/43395286?v=4?s=80" width="80px;" alt="BlueAmulet"/><br /><sub><b>BlueAmulet</b></sub></a><br /><a href="#ideas-BlueAmulet" title="Ideas, Planning, & Feedback">🤔</a> <a href="#question-BlueAmulet" title="Answering Questions">💬</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=BlueAmulet" title="Code">💻</a> <a href="#maintenance-BlueAmulet" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ThrowawayAccount01"><img src="https://avatars.githubusercontent.com/u/125531852?v=4?s=80" width="80px;" alt="ThrowawayAccount01"/><br /><sub><b>ThrowawayAccount01</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3AThrowawayAccount01" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MashiroSA"><img src="https://avatars.githubusercontent.com/u/40637516?v=4?s=80" width="80px;" alt="緋"/><br /><sub><b>緋</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=MashiroSA" title="Documentation">📖</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3AMashiroSA" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Lordmau5"><img src="https://avatars.githubusercontent.com/u/1345036?v=4?s=80" width="80px;" alt="Lordmau5"/><br /><sub><b>Lordmau5</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3ALordmau5" title="Bug reports">🐛</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=Lordmau5" title="Code">💻</a> <a href="#ideas-Lordmau5" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-Lordmau5" title="Maintenance">🚧</a> <a href="#question-Lordmau5" title="Answering Questions">💬</a> <a href="#userTesting-Lordmau5" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DL909"><img src="https://avatars.githubusercontent.com/u/71912115?v=4?s=80" width="80px;" alt="DL909"/><br /><sub><b>DL909</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3ADL909" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Satisfy256"><img src="https://avatars.githubusercontent.com/u/101394399?v=4?s=80" width="80px;" alt="Satisfy256"/><br /><sub><b>Satisfy256</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3ASatisfy256" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pierluigizagaria"><img src="https://avatars.githubusercontent.com/u/57801386?v=4?s=80" width="80px;" alt="Pierluigi Zagaria"/><br /><sub><b>Pierluigi Zagaria</b></sub></a><br /><a href="#userTesting-pierluigizagaria" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ruckusmattster"><img src="https://avatars.githubusercontent.com/u/77196088?v=4?s=80" width="80px;" alt="ruckusmattster"/><br /><sub><b>ruckusmattster</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Aruckusmattster" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Desuka-art"><img src="https://avatars.githubusercontent.com/u/111822082?v=4?s=80" width="80px;" alt="Desuka-art"/><br /><sub><b>Desuka-art</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3ADesuka-art" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/heyfixit"><img src="https://avatars.githubusercontent.com/u/41658450?v=4?s=80" width="80px;" alt="heyfixit"/><br /><sub><b>heyfixit</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=heyfixit" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.youtube.com/c/NerdyRodent"><img src="https://avatars.githubusercontent.com/u/74688049?v=4?s=80" width="80px;" alt="Nerdy Rodent"/><br /><sub><b>Nerdy Rodent</b></sub></a><br /><a href="#video-nerdyrodent" title="Videos">📹</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xieyumc"><img src="https://avatars.githubusercontent.com/u/47858007?v=4?s=80" width="80px;" alt="谢宇"/><br /><sub><b>谢宇</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=xieyumc" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ColdCawfee"><img src="https://avatars.githubusercontent.com/u/79474598?v=4?s=80" width="80px;" alt="ColdCawfee"/><br /><sub><b>ColdCawfee</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3AColdCawfee" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sbersier"><img src="https://avatars.githubusercontent.com/u/34165937?v=4?s=80" width="80px;" alt="sbersier"/><br /><sub><b>sbersier</b></sub></a><br /><a href="#ideas-sbersier" title="Ideas, Planning, & Feedback">🤔</a> <a href="#userTesting-sbersier" title="User Testing">📓</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Asbersier" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Meldoner"><img src="https://avatars.githubusercontent.com/u/43951115?v=4?s=80" width="80px;" alt="Meldoner"/><br /><sub><b>Meldoner</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3AMeldoner" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mmodeusher"><img src="https://avatars.githubusercontent.com/u/46575920?v=4?s=80" width="80px;" alt="mmodeusher"/><br /><sub><b>mmodeusher</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Ammodeusher" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AlonDan"><img src="https://avatars.githubusercontent.com/u/21152334?v=4?s=80" width="80px;" alt="AlonDan"/><br /><sub><b>AlonDan</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3AAlonDan" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Likkkez"><img src="https://avatars.githubusercontent.com/u/44336181?v=4?s=80" width="80px;" alt="Likkkez"/><br /><sub><b>Likkkez</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3ALikkkez" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DuctTapeGames"><img src="https://avatars.githubusercontent.com/u/84365142?v=4?s=80" width="80px;" alt="Duct Tape Games"/><br /><sub><b>Duct Tape Games</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3ADuctTapeGames" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://tec.hxlxz.com/"><img src="https://avatars.githubusercontent.com/u/6624983?v=4?s=80" width="80px;" alt="Xianglong He"/><br /><sub><b>Xianglong He</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Ahxl9654" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/75aosu"><img src="https://avatars.githubusercontent.com/u/79185331?v=4?s=80" width="80px;" alt="75aosu"/><br /><sub><b>75aosu</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3A75aosu" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tonyco82"><img src="https://avatars.githubusercontent.com/u/56610534?v=4?s=80" width="80px;" alt="tonyco82"/><br /><sub><b>tonyco82</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Atonyco82" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yxlllc"><img src="https://avatars.githubusercontent.com/u/33565655?v=4?s=80" width="80px;" alt="yxlllc"/><br /><sub><b>yxlllc</b></sub></a><br /><a href="#ideas-yxlllc" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=yxlllc" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/outhipped"><img src="https://avatars.githubusercontent.com/u/116147475?v=4?s=80" width="80px;" alt="outhipped"/><br /><sub><b>outhipped</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Aouthipped" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/escoolioinglesias"><img src="https://avatars.githubusercontent.com/u/73505402?v=4?s=80" width="80px;" alt="escoolioinglesias"/><br /><sub><b>escoolioinglesias</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Aescoolioinglesias" title="Bug reports">🐛</a> <a href="#userTesting-escoolioinglesias" title="User Testing">📓</a> <a href="#video-escoolioinglesias" title="Videos">📹</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Blacksingh"><img src="https://avatars.githubusercontent.com/u/130872856?v=4?s=80" width="80px;" alt="Blacksingh"/><br /><sub><b>Blacksingh</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3ABlacksingh" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://tybantarnusa.com"><img src="https://avatars.githubusercontent.com/u/9532857?v=4?s=80" width="80px;" alt="Mgs. M. Thoyib Antarnusa"/><br /><sub><b>Mgs. M. Thoyib Antarnusa</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Atybantarnusa" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ZeroHackz"><img src="https://avatars.githubusercontent.com/u/15729496?v=4?s=80" width="80px;" alt="Exosfeer"/><br /><sub><b>Exosfeer</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3AZeroHackz" title="Bug reports">🐛</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=ZeroHackz" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/guranon"><img src="https://avatars.githubusercontent.com/u/130421189?v=4?s=80" width="80px;" alt="guranon"/><br /><sub><b>guranon</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/issues?q=author%3Aguranon" title="Bug reports">🐛</a> <a href="#ideas-guranon" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=guranon" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alexanderkoumis"><img src="https://avatars.githubusercontent.com/u/5108856?v=4?s=80" width="80px;" alt="Alexander Koumis"/><br /><sub><b>Alexander Koumis</b></sub></a><br /><a href="https://github.com/voicepaw/so-vits-svc-fork/commits?author=alexanderkoumis" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
