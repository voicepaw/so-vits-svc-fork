# Installation

## One click easy installation

<a href="https://github.com/34j/so-vits-svc-fork/releases/download/v1.3.2/install.bat" download>
  <img src="https://img.shields.io/badge/.bat-download-blue?style=flat-square&logo=windows" alt="Download .bat">
</a>

## Install manually

### Creating a virtual environment

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

Installing without creating a virtual environment may cause a `PermissionError` if Python is installed in Program Files, etc.

### Install

Install this via pip (or your favourite package manager that uses pip):

```shell
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -U so-vits-svc-fork
```

- If no GPU is available, simply remove `pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu117`.
- If you are using an AMD GPU on Linux, replace `--index-url https://download.pytorch.org/whl/cu117` with `--index-url https://download.pytorch.org/whl/rocm5.4.2`. AMD GPUs are not supported on Windows (#120).
- If `fairseq` raises an error:
  - If it prompts [`Microsoft C++ Build Tools`](https://visualstudio.microsoft.com/visual-cpp-build-tools/) is not installed. please install it.
  - If it prompts that some dll is missing, reinstalling `Microsoft Visual C++ 2022` and `Windows SDK` may help.

### Update

Please update this package regularly to get the latest features and bug fixes.

```shell
pip install -U so-vits-svc-fork
```
