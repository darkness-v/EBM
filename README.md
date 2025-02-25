# EBM
This repository contains official pytorch implementation for following paper:
  - Title : Enhancing Audio Deepfake Detection by Improving Representation Similarity of Bonafide Speech
  - Autor : Seung-bin Kim, Hyun-seo Shin, Jungwoo Heo, Chan-yeong Lim, Kyo-Won Koo, Jisoo Son, Sanghyun Hong, Souhwan Jung, and Ha-Jin Yu

# Abstract
![Image](https://github.com/user-attachments/assets/4de70d04-edc3-4290-8521-5d52c0a00f1c)

The key to audio deepfake detection is distinguishing bonafide speech from carefully generated spoofed speech.
The more distinguishable they are, the better and more generalizable the detection becomes.
In this work, we propose a novel approach to enhance this distinguishability in the latent space.
Inspired by one-class classification, we formulate an objective function that encourages the contraction of bonafide samples while dispersing fake speech samples during training.
Our objective consists of two key components: Bonafide-Pair Learning (BPL) loss and an Extended One-Class Softmax (EOC-S) loss.
The BPL reduces intra-class variance by aligning the embeddings of augmented bonafide pairs, while the EOC-S leverages Adam-based centroid updates and margin constraints to reinforce separability from spoofed data.
Experimental results on ASVspoof datasets demonstrate that our proposed approach enhances detection performance across diverse attack scenarios.

Our experimental code was modified based on [here](https://github.com/talkingnow/HM-Conformer).


# Data
The [ASVspoof 2019 LA](https://www.asvspoof.org/index2019.html) and [ASVspoof 2021](https://www.asvspoof.org/index2021.html) datasets were used for training and test.
The ASVspoof 2019 LA trainset consists of 2,580 bona fide samples and 22,800 spoof samples.

Additionally, we applied vocoder augmentation to the training set using [HiFi-GAN](https://arxiv.org/pdf/2010.05646).
The vocoder was applied only to spoof samples and the method for applying the HiFi-GAN is described [here](https://github.com/jik876/hifi-gan).


# Environment
Docker image (nvcr.io/nvidia/pytorch:23.08-py3) of Nvidia GPU Cloud was used for conducting our experiments.

Make docker image and activate docker container.
```
.docker_build.sh
.docker_run.sh
```

Note that you need to modify the mapping path (/data) before running the 'docker_run.sh' file.
Additionally, the dataset path is specified in the `arguments.py` file, so it must be set accordingly:

In `arguments.py`
```
'path_19LA'    : '/data/ASVspoof2019',
'path_21LA'  : '/data/ASVspoof2021_LA_eval',
'path_21DF'  : '/data/ASVspoof2021_DF'
```

We have a basic logger that stores information in local. However, if you would like to use an additional online neptune logger:
```
# Neptune: Add 'neptune_user' and 'neptune_token'
# input this arguments in "system_args" dictionary:
# for example
'neptune_user'  : 'user-name',
'neptune_token' : 'NEPTUNE_TOKEN'
```
        

# Training
- Trining on a single GPU
```
CUDA_VISIBLE_DEVICES=0 python3 main.py
```
- Trining on multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py
```


# Citation
Please cite if you make use of the code.

```
@article{kim2025ebm,
  title={Enhancing Audio Deepfake Detection by Improving Representation Similarity of Bonafide Speech},
  author={Seung-bin Kim, Hyun-seo Shin, Jungwoo Heo, Chan-yeong Lim, Kyo-Won Koo, Jisoo Son, Sanghyun Hong, Souhwan Jung, and Ha-Jin Yu},
  journal={},
  year={2025}
}