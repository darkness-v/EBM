FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN apt-get update

RUN pip install pip --upgrade

RUN pip install torch==2.1.0a0+29c30b1
RUN pip install torchaudio==2.0.1
RUN pip install neptune-client
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install huggingface_hub
RUN pip uninstall transformer-engine -y

RUN pip install torchsummary
RUN pip install torch-audiomentations

RUN apt-get install -y git-lfs
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install ffmpeg-python
RUN pip install pydub


