@echo off

echo 本bat汉化基于英文版，对原版进行了一些本地工作和优化，如安装过程有问题，可以尝试安装原版
echo.

echo.
echo 检查 Python 版本 3.10...
echo.

py -3.10 --version >nul 2>&1
if %errorlevel%==0 (
    echo Python 3.10 已经安装
	echo.
) else (
    echo Python 3.10 未安装，开始下载...
	echo.
    curl https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe -o python-3.10.10-amd64.exe

    echo 安装 Python 3.10...
	echo.
    python-3.10.10-amd64.exe /quiet InstallAllUsers=1 PrependPath=1

    echo 清理安装器...
	echo.
    del python-3.10.10-amd64.exe
)
echo.
echo 检查 GPU...
echo.
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo 找到可用GPU
	echo.
) else (
    echo 未找到可用found
	echo.
)

nvidia-smi >nul 2>&1
if %errorlevel%==0 (

	echo.
    echo 检查CUDA...
	echo.

    if %errorlevel%==0 (
        echo CUDA 已经安装
		echo.
    ) else (
        echo 未检测到CUDA，请从下面链接手动安装CUDA，安装后再重新运行本程序
		echo https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows
		echo.
		echo 如果你已经确定安装了CUDA，可能是程序检测出错，可以按任意键强制继续执行，否则请关闭本程序，安装好CUDA后再重新运行
		echo.
		Pause
    )

    echo 检查 cuDNN...
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll" (
        echo cuDNN 已经安装
		echo.
    ) else (
        echo 未检测到cuDNN，请从下面链接手动安装CUDA，安装后再重新运行本程序
		echo https://developer.nvidia.com/cudnn (https://developer.nvidia.com/downloads/compute/cudnn/secure/8.8.1/local_installers/11.8/cudnn-windows-x86_64-8.8.1.3_cuda11-archive.zip/)
		echo.
		echo 如果你已经确定安装了cuDNN，可能是程序检测出错，可以按任意键强制继续执行，否则请关闭本程序，安装好CUDA后再重新运行
		echo.
		Pause
    )
)
echo.
echo 正在创建虚拟环境（需要一点时间，请耐心等待）...
echo.
py -3.10 -m venv venv
echo.
echo 升级 pip 和 wheel...
echo.
venv\Scripts\python.exe -m pip install --upgrade pip wheel
echo.
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
echo 安装 PyTorch （GPU版本）...
echo.
venv\Scripts\pip.exe install torch torchvision torchaudio --index-url  https://mirror.sjtu.edu.cn/pytorch-wheels
    echo 安装 PyTorch （CPU版本）...
	echo.
    venv\Scripts\pip.exe install torch torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple pyspider
)
echo.
echo 请检查以上包是否都成功安装，确定成功安装后，按任意键开始安装so-vits-svc-fork
echo.
Pause
echo 安装 so-vits-svc-fork...
echo.
venv\Scripts\pip.exe install so-vits-svc-fork
echo.
echo 启动 so-vits-svc-fork 图形化界面...
echo.
venv\Scripts\svcg.exe

Pause
