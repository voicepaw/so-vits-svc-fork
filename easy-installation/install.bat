@echo off

echo Checking for Python 3.10...

py -3.10 --version >nul 2>&1
if %errorlevel%==0 (
    echo Python 3.10 is already installed.
) else (
    echo Python 3.10 is not installed. Downloading installer...
    curl https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe -o python-3.10.10-amd64.exe

    echo Installing Python 3.10...
    python-3.10.10-amd64.exe /quiet InstallAllUsers=1 PrependPath=1

    echo Cleaning up installer...
    del python-3.10.10-amd64.exe
)

echo Checking GPU...
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set GPU=1
) else (
    set GPU=0
)

if %GPU%==1 (
    echo Checking CUDA...
    nvcc --version
    if %errorlevel%==0 (
        echo CUDA is already installed.
    ) else (
        echo Please install CUDA 11.7 from https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows & exit /b 1
    )

    echo Checking cuDNN...
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin\cudnn64_8.dll" (
        echo cuDNN is already installed.
    ) else (
        echo Please install cuDNN for CUDA 11.x from https://developer.nvidia.com/cudnn (https://developer.nvidia.com/downloads/compute/cudnn/secure/8.8.1/local_installers/11.8/cudnn-windows-x86_64-8.8.1.3_cuda11-archive.zip/) & exit /b 1
    )
)

echo Creating virtual environment...
py -3.10 -m venv venv

echo Updating pip and wheel...
venv\Scripts\python.exe -m pip install --upgrade pip wheel

if %GPU%==1 (
    echo Installing PyTorch with GPU support...
    venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cu117
) else (
    echo Installing PyTorch without GPU support...
    venv\Scripts\pip.exe install torch torchaudio
)

echo Installing so-vits-svc-fork...
venv\Scripts\pip.exe install so-vits-svc-fork

echo Launching so-vits-svc-fork GUI...
venv\Scripts\svcg.exe
