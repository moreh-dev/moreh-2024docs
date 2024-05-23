---
icon: terminal
tags: [guide]
order: 800
---

# GPT Fine-tuning

이 튜토리얼은 MoAI Platform에서 [Hugging Face](https://huggingface.co)에 오픈소스로 공개된 GPT 기반의 모델을 fine-tuning하는 예시를 소개합니다. 튜토리얼을 통해 MoAI Platform으로 AMD GPU 클러스터를 사용하는 방법을 익히고 성능 및 자동 병렬화의 이점을 확인할 수 있습니다.

## 개요

GPT는 Transformer decoder 구조만을 사용한 언어 모델 아키텍처로써, 2018년 [OpenAI](https://openai.com/)에서 GPT-1을 통해 처음 공개하였습니다. 이후 OpenAI에서는 사전학습에 사용되는 데이터셋 크기와 모델 파라미터를 늘려가며 GPT-2를 거쳐 GPT-3, GPT-4 모델을 개발했으며 이 중 오픈소스로 공개된 모델은 GPT-1, GPT-2 입니다.  

GPT의 기본적인 아키텍처는 오픈소스로 공개되어 있기 때문에 Huggingface에서는 OpenAI에서 개발한 모델 이외에도 다양한 GPT 기반의 모델을 찾아볼 수 있습니다. 

이 튜토리얼에서는 코드 생성 태스크에 대해 MoAI Platform을 활용해 [Cerebras-GPT-13B](https://huggingface.co/cerebras/Cerebras-GPT-13B) 모델을 fine-tuning 해보겠습니다.  

## 시작하기 전에

MoAI Platform 상의 컨테이너 혹은 가상 머신을 인프라 제공자로부터 발급받고, 여기에 SSH로 접속하는 방법을 안내 받으시기 바랍니다. 예를 들어 MoAI Platform 기반으로 운영되는 다음 퍼블릭 클라우드 서비스를 신청하여 사용할 수 있습니다.

- KT Cloud의 Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

혹은 일시적으로 체험판 컨테이너 및 GPU 자원을 할당 받기를 원하시는 분은 Moreh(support@moreh.io)에 문의하시기 바랍니다.

SSH로 접속한 다음 `moreh-smi` 명령을 실행하여 MoAI Accelerator가 잘 표시되는지 확인하시기 바랍니다. 디바이스 이름은 시스템마다 다르게 설정되어 있을 수 있습니다. 

### MoAI Accelerator 확인

이 튜토리얼에서 안내할 GPT 모델과 같은 sLLM을 학습하기 위해서는 적절한 크기의 MoAI Accelerator를 선택해야 합니다. 먼저 `moreh-smi` 명령어를 이용해 현재 사용중인 MoAI Accelerator를 확인합니다. 

수행할 학습에 필요한 구체적인 MoAI Accelerator 설정에 대한 설명은 [3. 학습 실행하기](3_학습_실행하기.md)에서 제공하겠습니다.  

```bash
$ moreh-smi
11:40:36 April 16, 2024
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.2.0  Latest Version: 24.2.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  Large.256GB  |  -             |  -             |  -           |
+-------------------------------------------------------------------------------------------------+
```

