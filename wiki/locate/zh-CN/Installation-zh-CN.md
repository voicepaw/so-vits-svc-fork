# 安装教程

## 可以使用 bat 一键安装

<a href="https://github.com/34j/so-vits-svc-fork/releases/download/v1.3.2/install.bat" download>
  <img src="https://img.shields.io/badge/.bat-download-blue?style=flat-square&logo=windows" alt="Download .bat">
</a>

## 手动安装

### 创建虚拟环境

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

不创建 Venv 虚拟环境而直接安装可能会导致 `PermissionError` 如果`Python`是安装在`Program Files`等情况.

### 安装

通过 pip 安装 (或者通过包管理器使用 pip 安装):

```shell
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -U so-vits-svc-fork
```

-   如果没有可用 GPU, 不需要执行 `pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu117`.
-   如果在 Linux 下使用 AMD GPU, 请使用此命令 `--index-url https://download.pytorch.org/whl/rocm5.4.2`
    替换掉 `--index-url https://download.pytorch.org/whl/cu117` . Windows 下不支持 AMD GPUs (#120).
-   如果 `fairseq` 报错:
    -   如果提示 [`Microsoft C++ Build Tools`](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 没有安装. 安装即可.
    -   如果提示缺少 dll 文件, 重新安装 `Microsoft Visual C++ 2022` 和 `Windows SDK` 可能有用

### 更新

请经常更新以获取最新功能和修复错误:

```shell
pip install -U so-vits-svc-fork
```
