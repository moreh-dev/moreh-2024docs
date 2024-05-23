---
icon: terminal
tags: [guide]
order: 40
---

# 1. Fine-tuning 준비하기

MoAI Platform에서 PyTorch 스크립트 실행 환경을 준비하는 것은 일반적인 GPU 서버에서와 크게 다르지 않습니다.

## PyTorch 설치 여부 확인하기

SSH로 컨테이너에 접속한 다음 아래와 같이 실행하여 현재 conda 환경에 PyTorch가 설치되어 있는지 확인합니다.

```bash
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.2.0          pypi_0    pypi
...
```

버전명에는 PyTorch 버전과 이를 실행시키기 위한 MoAI 버전이 함께 표시되어 있습니다. 위 예시의 경우 PyTorch 1.13.1+cu116 버전을 실행하는 MoAI의 24.2.0 버전이 설치되어 있음을 의미합니다.

만약 `conda: command not found` 메시지가 표시되거나, torch 패키지가 리스트되지 않거나, 혹은 torch 패키지가 존재하더라도 버전명에 “moreh”가 포함되지 않은 경우 ***([Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md))*** 문서에 따라 conda 환경을 생성하십시오.

## PyTorch 동작 여부 확인하기

다음과 같이 실행하여 torch 패키지가 정상적으로 import되고 MoAI Accelerator가 인식되는지 확인합니다.

```bash
$ python
Python 3.8.18 (default, Sep 11 2023, 13:40:15)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
...
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name()
[2024-04-16 19:17:45.714] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-16 19:17:45.752] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-16 19:17:47.409] [info] [1/1] Connecting to resources on the server (192.168.110.00:24158)...
[2024-04-16 19:17:47.452] [info] Establishing links to the resources...
[2024-04-16 19:17:47.636] [info] MoAI Accelerator is ready to use.
'MoAI Accelerator'
>>> quit()
```

## 필요 Python 패키지 설치

다음과 같이 실행하여 스크립트 실행에 필요한 서드 파티 Python 패키지들을 미리 설치합니다.

```bash
$ pip install transformers==4.34.0 datasets==2.14.5 loguru==0.7.2
```

## 학습 스크립트 다운로드

다음과 같이 실행하여 GitHub 레포지토리에서 학습을 위한 PyTorch 스크립트를 다운로드합니다. 본 튜토리얼에서는 `tutorial` 디렉토리 안에 있는 `train_gpt.py` 스크립트를 사용할 것입니다.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_gpt.py  ...
```

## 학습 데이터 다운로드

Huggingface에서는 model checkpoint뿐만 아니라 model fine-tuning에 사용할 수 있는 다양한 데이터셋이 공개되어 있습니다. 

이번 튜토리얼에서 저희는 [mlabonne/Evol-Instruct-Python-26k](https://huggingface.co/datasets/mlabonne/Evol-Instruct-Python-26k) 데이터셋을 사용할 것입니다. 이 데이터셋은 주어진 질문 조건과 주어진 질문 조건에 맞게 작성된 python 코드로 이루어져 있습니다.

학습 데이터를 다운로드 받기 위해 `dataset` 디렉토리 안에 있는`prepare_gpt_dataset.py` 스크립트를 통해 huggingface에 공개된 데이터셋을 다운로드하고, fine-tuning 학습에 바로 사용할 수 있도록 전처리를 진행하겠습니다.

```bash
~/quickstart$ python dataset/prepare_gpt_dataset.py
```

전처리가 진행된 데이터셋은 `gpt_dataset.pt` 로 저장됩니다. 

저장된 데이터셋은 코드상에서 다음과 같이 로드하여 사용할 수 있습니다.

```python
dataset = torch.load("gpt_dataset.pt")
```

