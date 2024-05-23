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
$ pip install transformers==4.39.3 datasets==2.19.1 loguru==0.7.2
```

## 학습 스크립트 다운로드

다음과 같이 실행하여 GitHub 레포지토리에서 학습을 위한 PyTorch 스크립트를 다운로드합니다. 본 튜토리얼에서는 `tutorial` 디렉토리 안에 있는 `train_mistral.py` 스크립트를 사용할 것입니다.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_mistral.py  ...
```

## 학습 모델 및 토크나이저 다운로드

Hugging Face를 이용해 Mistral 7B v0.1 모델의 체크포인트와 토크나이저를 다운로드 받습니다. 이때 Mistral 모델은 커뮤니티 라이센스 동의와 Hugging Face 토큰 정보가 필요합니다. 또한 Mistral 7B 모델의 경우 체크포인트 용량이 약 15GB이기 때문에 체크포인트를 위한 16GB 이상의 스토리지 여유가 권장됩니다.

먼저 다음 사이트에서 필요한 정보를 입력한 후 라이센스 동의를 진행합니다.

[!ref icon="link-external" text="mistralai/Mistral-7B-v0.1 · Hugging Face"](https://huggingface.co/mistralai/Mistral-7B-v0.1)

동의서 제출 후 페이지의 상태가 다음과 같이 변경된 것을 확인합니다.

![](alert.png)

상태 변경이 되었다면, 다음과 같이 `tutorial` 디렉토리 안의 `download_mistral_7b.py` 스크립트를 이용해 모델 체크포인트와 토크나이저를 `./mistral-7b` 디렉토리에 다운로드 받을 수 있습니다. 

`<user-token>` 은 사용자의 Hugging Face 토큰으로 치환합니다.

```bash
~/quickstart$ python tutorial/download_mistral_7b.py --token <user-token>
```

모델 체크포인트와 토크나이저가 다운로드 받아졌는지 확인합니다.

```bash
~/quickstart$ ls ./mistral-7b
config.json                       model-00003-of-00006.safetensors  model.safetensors.index.json  tokenizer.model
generation_config.json            model-00004-of-00006.safetensors  special_tokens_map.json
model-00001-of-00006.safetensors  model-00005-of-00006.safetensors  tokenizer_config.json
model-00002-of-00006.safetensors  model-00006-of-00006.safetensors  tokenizer.json
```

## 학습 데이터 다운로드

이 튜토리얼에서는 코드 생성 훈련을 위해 공개된 여러 데이터셋들 중 Hugging Face에 공개되어 있는 [python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) 데이터셋(11.4 MB)을 사용할 것입니다.

`prepare_mistral_dataset.py` 를 실행해 데이터셋을 다운로드하고, 학습에 사용할 수 있도록 전처리를 진행합니다.

```
~/quickstart$ ls dataset
...  prepare_mistral_dataset.py ...

~/quickstart$ python dataset/dataset_qwen.py
2024-04-19 03:27:05,865 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpjkaqeu3r
2024-04-19 03:27:05,866 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpjkaqeu3r/_remote_module_non_scriptable.py
2024-04-19 03:27:24,010 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
Loading Tokenizer...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Downloading dataset...
Preprocessing dataset...
Saving datset into torch format...
Dataset saved as ./mistral_dataset.pt

~/quickstart$ ls
... mistral_dataset.pt ...
```

전처리가 진행된 데이터셋은 `mistral_dataset.pt` 로 저장됩니다. 

저장된 데이터셋은 코드상에서 다음과 같이 로드하여 사용할 수 있습니다.

```bash
dataset = torch.load("./mistral_dataset.pt")
```
