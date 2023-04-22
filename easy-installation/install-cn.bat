@echo off


echo.

echo.
echo ��� Python �汾 3.10...
echo.

py -3.10 --version >nul 2>&1
if %errorlevel%==0 (
    echo Python 3.10 �Ѿ���װ
	echo.
) else (
    echo Python 3.10 δ��װ����ʼ����...
	echo.
    curl https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe -o python-3.10.10-amd64.exe

    echo ��װ Python 3.10...
	echo.
    python-3.10.10-amd64.exe /quiet InstallAllUsers=1 PrependPath=1

    echo ����װ��...
	echo.
    del python-3.10.10-amd64.exe
)
echo.
echo ��� GPU...
echo.
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo �ҵ�����GPU
	echo.
) else (
    echo δ�ҵ�����found
	echo.
)

nvidia-smi >nul 2>&1
if %errorlevel%==0 (

	echo.
    echo ���CUDA...
	echo.

    if %errorlevel%==0 (
        echo CUDA �Ѿ���װ
		echo.
    ) else (
        echo δ��⵽CUDA��������������ֶ���װCUDA����װ�����������б�����
		echo https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows
		echo.
		echo ������Ѿ�ȷ����װ��CUDA�������ǳ�����������԰������ǿ�Ƽ���ִ�У�������رձ����򣬰�װ��CUDA������������
		echo.
		Pause
    )

    echo ��� cuDNN...
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll" (
        echo cuDNN �Ѿ���װ
		echo.
    ) else (
        echo δ��⵽cuDNN��������������ֶ���װCUDA����װ�����������б�����
		echo https://developer.nvidia.com/cudnn (https://developer.nvidia.com/downloads/compute/cudnn/secure/8.8.1/local_installers/11.8/cudnn-windows-x86_64-8.8.1.3_cuda11-archive.zip/)
		echo.
		echo ������Ѿ�ȷ����װ��cuDNN�������ǳ�����������԰������ǿ�Ƽ���ִ�У�������رձ����򣬰�װ��CUDA������������
		echo.
		Pause
    )
)
echo.
echo ���ڴ������⻷������Ҫһ��ʱ�䣬�����ĵȴ���...
echo.
py -3.10 -m venv venv
echo.
echo ���� pip �� wheel...
echo.
venv\Scripts\python.exe -m pip install --upgrade pip wheel
echo.
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
echo ��װ PyTorch ��GPU�汾��...
echo.
venv\Scripts\pip.exe install torch torchvision torchaudio --index-url  https://mirror.sjtu.edu.cn/pytorch-wheels
    echo ��װ PyTorch ��CPU�汾��...
	echo.
    venv\Scripts\pip.exe install torch torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple pyspider
)
echo.
echo �������ϰ��Ƿ񶼳ɹ���װ��ȷ���ɹ���װ�󣬰��������ʼ��װso-vits-svc-fork
echo.
Pause
echo ��װ so-vits-svc-fork...
echo.
venv\Scripts\pip.exe install so-vits-svc-fork
echo.
echo ���� so-vits-svc-fork ͼ�λ�����...
echo.
venv\Scripts\svcg.exe

Pause
