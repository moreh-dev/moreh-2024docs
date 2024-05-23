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

만약 `conda: command not found` 메시지가 표시되거나, torch 패키지가 리스트되지 않거나, 혹은 torch 패키지가 존재하더라도 버전명에 “moreh”가 포함되지 않은 경우 **[Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md)** 문서에 따라 conda 환경을 생성하십시오.

만약 해당 MoAI 버전이 24.2.0이 아닌 다른 버전이라면 아래의 코드를 실행시키십시오.

```bash
$ update-moreh --target 24.2.0
Currently installed: 24.3.0
Possible upgrading version: 24.2.0

Do you want to upgrade? (y/n, default:n)
y
```

## PyTorch 동작 여부 확인하기

다음과 같이 실행하여 torch 패키지가 정상적으로 import 되고 MoAI Accelerator가 인식되는지 확인합니다.

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
$ pip install transformers==4.36.2 datasets==2.19.0 loguru==0.7.2 sentencepiece==0.2.0
```

## 학습 스크립트 다운로드

다음과 같이 실행하여 GitHub 레포지토리에서 학습을 위한 PyTorch 스크립트를 다운로드합니다. 본 튜토리얼에서는 `tutorial` 디렉토리 안에 있는 `train_llama2.py` 스크립트를 사용할 것입니다.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_llama2.py  ...
```

## 필요 Python 패키지 설치

다음과 같이 실행하여 스크립트 실행에 필요한 서드 파티 Python 패키지들을 미리 설치합니다.

```bash
$ pip install -r requirements/requirements_baichuan.txt
```

## 학습 모델 및 토크나이저 다운로드

Hugging Face를 이용해 Llama2-13b-hf 모델의 체크포인트와 토크나이저를 다운로드합니다. 이때 Llama2 모델은 커뮤니티 라이센스 동의와 Hugging Face 토큰 정보가 필요합니다. 또한 Llama2 13B 모델의 경우 체크포인트 용량이 약 49GB이기 때문에 체크포인트를 위한 50GB 스토리지 여유가 필수적입니다.

먼저 다음 사이트에서 필요한 정보를 입력한 후 라이센스 동의를 진행합니다.

[meta-llama/Llama-2-13b-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-13b-hf)

동의서 제출 후 페이지의 상태가 다음과 같이 변경된 것을 확인합니다.

![](alert.png)

상태 변경이 되었다면, 다음과 같이 `tutorial` 디렉토리 안의 `download_llama2_13b.py` 스크립트를 이용해 모델 체크포인트와 토크나이저를 `./llama-2-13b-hf` 디렉토리에 다운로드 받을 수 있습니다. 

`<user-token>` 은 사용자의 Hugging Face 토큰으로 치환합니다.

```bash
~/quickstart$ python tutorial/download_llama2_13b.py --token <user-token>
```

모델 체크포인트와 토크나이저가 다운로드 받아졌는지 확인합니다.

```bash
~/quickstart$ ls ./llama-2-13b-hf
config.json                       model-00008-of-00011.safetensors
generation_config.json            model-00009-of-00011.safetensors
model-00001-of-00011.safetensors  model-00010-of-00011.safetensors
model-00002-of-00011.safetensors  model-00011-of-00011.safetensors
model-00003-of-00011.safetensors  model.safetensors.index.json
model-00004-of-00011.safetensors  special_tokens_map.json
model-00005-of-00011.safetensors  tokenizer_config.json
model-00006-of-00011.safetensors  tokenizer.json
model-00007-of-00011.safetensors  tokenizer.model
```

## 학습 데이터 다운로드

학습 데이터를 다운로드 받기 위해 `dataset` 디렉토리 안에 있는 `prepare_llama2_dataset.py` 스크립트를 사용하겠습니다. 코드를 실행하면 [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) 데이터를 다운로드 받고 학습에 사용할 수 있도록 전처리를 진행하여 `llama2_dataset.pt` 파일로 저장합니다.

```bash
~/quickstart$ ls dataset
...  prepare_llama2_dataset.py ...

~/quickstart$ python dataset/prepare_llama2_dataset.py
2024-04-19 03:27:05,865 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpjkaqeu3r
2024-04-19 03:27:05,866 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpjkaqeu3r/_remote_module_non_scriptable.py
2024-04-19 03:27:24,010 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
Loading Tokenizer...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Downloading dataset...
Preprocessing dataset...
Saving datset into torch format...
Dataset saved as ./llama2_dataset.pt

~/quickstart$ ls
... llama2_dataset.pt ...
```

저장된 데이터셋은 코드상에서 다음과 같이 로드하여 사용할 수 있습니다.

```bash
dataset = torch.load("./llama2_dataset.pt")
```