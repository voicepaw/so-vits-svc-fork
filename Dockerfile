FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN ["apt", "update"]
RUN ["apt", "install", "-y", "build-essential"]
RUN ["pip", "install", "-U", "pip", "setuptools", "wheel"]
RUN ["pip", "install", "-U", "so-vits-svc-fork"]
ENTRYPOINT [ "svcg" ]
