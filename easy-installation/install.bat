@echo off

echo You can rerun this script to update the installation.

echo Moving to AppData\Roaming\so-vits-svc-fork...
mkdir "%APPDATA%\so-vits-svc-fork" >nul 2>&1
cd "%APPDATA%\so-vits-svc-fork"

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

echo Creating virtual environment...
py -3.10 -m venv venv

echo Updating pip and wheel...
venv\Scripts\python.exe -m pip install --upgrade pip wheel

nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo Installing PyTorch with GPU support...
venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing PyTorch without GPU support...
    venv\Scripts\pip.exe install torch torchaudio
)

echo Installing so-vits-svc-fork...
venv\Scripts\pip.exe install so-vits-svc-fork

rem echo Creating shortcut...
rem powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%USDRPROFILE%\Desktop\so-vits-svc-fork.lnk');$s.TargetPath='%APPDATA%\so-vits-svc-fork\venv\Scripts\svcg.exe';$s.Save()"

echo Creating shortcut to the start menu...
powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%APPDATA%\Microsoft\Windows\Start Menu\Programs\so-vits-svc-fork.lnk');$s.TargetPath='%APPDATA%\so-vits-svc-fork\venv\Scripts\svcg.exe';$s.Save()"

echo Launching so-vits-svc-fork GUI...
venv\Scripts\svcg.exe
