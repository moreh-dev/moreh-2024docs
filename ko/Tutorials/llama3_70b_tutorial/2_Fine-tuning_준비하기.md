---
icon: terminal
tags: [tutorial, llama3_70b]
order: 40
---

# 2. Fine-tuning 준비하기

## 시작하기 전에

MoAI Platform 상의 컨테이너 혹은 가상 머신을 인프라 제공자로부터 발급받고, 여기에 SSH로 접속하는 방법을 안내 받으시기 바랍니다. 예를 들어 MoAI Platform의 체험판 컨테이너 또는 MoAI Platform 기반으로 운영되는 퍼블릭 클라우드 서비스를 신청하여 사용할 수 있습니다.

- MoAI Platform 체험판 컨테이너 사용 문의: [support@moreh.io](mailto:support@moreh.io)
- KT Cloud Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

SSH로 접속한 다음 `moreh-smi` 명령을 실행하여 MoAI Accelerator가 잘 표시되는지 확인하시기 바랍니다. 디바이스 이름은 시스템마다 다르게 설정되어 있을 수 있습니다.

### MoAI Accelerator 확인

이 튜토리얼에서 안내할 Llama3 모델과 같은 sLLM을 학습하기 위해서는 적절한 크기의 MoAI Accelerator를 선택해야 합니다. 먼저 `moreh-smi` 명령어를 이용해 현재 사용중인 MoAI Accelerator를 확인합니다. 

수행할 학습에 필요한 구체적인 MoAI Accelerator 설정에 대한 설명은 “3. 학습 실행하기”에서 제공하겠습니다.  

```bash
$ moreh-smi
+---------------------------------------------------------------------------------------------------+
|                                                  Current Version: 24.5.0  Latest Version: 24.5.0  |
+---------------------------------------------------------------------------------------------------+
|  Device  |        Name         |      Model     |  Memory Usage  |  Total Memory  |  Utilization  |
+===================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+---------------------------------------------------------------------------------------------------+
```

MoAI Platform에서 PyTorch 스크립트 실행 환경을 준비하는 것은 일반적인 GPU 서버에서와 크게 다르지 않습니다.

### PyTorch 설치 여부 확인하기

SSH로 컨테이너에 접속한 다음 아래와 같이 실행하여 현재 conda 환경에 PyTorch가 설치되어 있는지 확인합니다.

```bash
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.5.0          pypi_0    pypi
...
```

버전명에는 PyTorch 버전과 이를 실행시키기 위한 MoAI 버전이 함께 표시되어 있습니다. 위 예시의 경우 PyTorch 1.13.1+cu116 버전을 실행하는 MoAI의 24.5.0 버전이 설치되어 있음을 의미합니다.

만약 `conda: command not found` 메시지가 표시되거나, torch 패키지가 리스트되지 않거나, 혹은 torch 패키지가 존재하더라도 버전명에 “moreh”가 포함되지 않은 경우 **[MoAI Platform에서 Fine-tuning 준비하기](../../Supported_Documents/Prepare_Fine_tuning_MoAI.md)** 문서에 따라 conda 환경을 생성하십시오.

### PyTorch 동작 여부 확인하기

다음과 같이 실행하여 torch 패키지가 정상적으로 import되고 MoAI Accelerator가 인식되는지 확인합니다.

```bash
$ python
>>> import torch
...
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name()
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/1] Connecting to resources on the server (192.168.110.00:24158)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
'MoAI Accelerator'
>>> quit()
```

### 학습 스크립트 다운로드

다음과 같이 실행하여 GitHub 레포지토리에서 학습을 위한 PyTorch 스크립트를 다운로드합니다. 본 튜토리얼에서는 `tutorial` 디렉토리 안에 있는 `train_llama3.py` 스크립트를 사용할 것입니다.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_llama3.py  ...
```

### 필요 Python 패키지 설치

다음과 같이 실행하여 스크립트 실행에 필요한 서드 파티 Python 패키지들을 미리 설치합니다.

```bash
$ pip install -r requirements/requirements_llama3.txt
```

### 학습 모델 및 토크나이저 다운로드

Hugging Face에 공개된 Llama3 70B 모델 체크포인트를 사용하기 위해서는 커뮤니티 라이센스 동의와 Hugging Face 토큰 정보가 필요합니다.

먼저 다음 사이트에서 필요한 정보를 입력한 후 라이센스 동의를 진행합니다.

동의서 제출 후 페이지의 상태가 다음과 같이 변경된 것을 확인합니다.

[!ref icon="link-external" text="meta-llama/Meta-Llama-3-70B · Hugging Face"](https://huggingface.co/meta-llama/Meta-Llama-3-70B)

동의서 제출 후 페이지의 상태가 다음과 같이 변경된 것을 확인합니다.

![](alert.png)

다음과 같은 명령어를 터미널에 입력하고 안내에 따라 Hugging Face 사용자 토큰을 입력합니다. 

```bash
huggingface-cli login
```
