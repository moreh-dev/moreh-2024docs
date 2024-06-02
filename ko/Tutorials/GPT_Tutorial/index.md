---
icon: terminal
tags: [tutorial, gpt]
order: 800
---

# GPT Fine-tuning

이 튜토리얼은 MoAI Platform에서 [Hugging Face](https://huggingface.co)에 오픈소스로 공개된 GPT 기반의 모델을 fine-tuning하는 예시를 소개합니다. 튜토리얼을 통해 아래와 같은 MoAI Platform이 제공하는 여러 기능을 체험하며, AMD GPU 클러스터를 사용하는 방법을 익힐 수 있습니다.

- 사용자는 수십 개의 GPU를 MoAI Accelerator라는 하나의 가속기처럼 사용할 수 있어 복잡한 병렬화 작업이나 클러스터 환경 설정 없이도 쉽게 학습을 실행할 수 있습니다. 사용자는 리소스 관리에 신경쓰지 않고 학습에만 집중할 수 있습니다.
- 자동 병렬화 기능 덕분에 코드 작성과 개발이 간소화되며, 모델 학습 속도가 크게 향상됩니다. 이는 효율적인 자원 활용을 가능하게 하여 사용자가 더 빠르고 효과적으로 작업할 수 있도록 돕습니다.

## 개요

MoAI Platform은 수천 대의 GPU를 쉽게 제어하여 AI 모델을 학습하거나 추론할 수 있는 확장 가능한 AI 플랫폼입니다. MoAI Platform의 특징은 모델을 fine-tuning할 때 가상화와 병렬화를 통해 매우 간단한 학습 방법을 제공한다는 점입니다.

MoAI Platform은 여러개의 GPU를 가상화하여 하나의 가속기인 [MoAI Accelerator](https://docs.moreh.io/ko/moai_features/virtualization/#gpu-%EA%B0%80%EC%83%81%ED%99%94-moai-accelerator)로 제공합니다. 따라서 다중 GPU 사용을 위해 필요한 사전 준비나 코드 수정이 필요하지 않습니다.

MoAI Platform은 고객이 가상화된 MoAI Accelerator를 사용할 때 내부적으로 자동 최적화된 병렬화를 제공합니다. 모델 크기, 데이터 크기에 대해서 다양한 병렬화 방법을 고려해 최적의 병렬화 환경을 제공하며, 사용자는 별도의 작업이 없이 간단한 코드로 고성능 학습을 경험할 수 있습니다.

## 시작하기 전에

MoAI Platform 상의 컨테이너 혹은 가상 머신을 인프라 제공자로부터 발급받고, 여기에 SSH로 접속하는 방법을 안내 받으시기 바랍니다. 예를 들어 MoAI Platform의 체험판 컨테이너 또는 MoAI Platform 기반으로 운영되는 퍼블릭 클라우드 서비스를 신청하여 사용할 수 있습니다.

- MoAI Platform 체험판 컨테이너 사용 문의: [support@moreh.io](mailto:support@moreh.io)
- [KT Cloud Hyperscale AI Computing](https://cloud.kt.com/solution/hyperscaleAiComputing/)

SSH로 접속한 다음 `moreh-smi` 명령을 실행하여 MoAI Accelerator가 잘 표시되는지 확인하시기 바랍니다. 디바이스 이름은 시스템마다 다르게 설정되어 있을 수 있습니다.

### MoAI Accelerator 확인

이 튜토리얼에서 안내할 GPT 모델과 같은 sLLM을 학습하기 위해서는 적절한 크기의 MoAI Accelerator를 선택해야 합니다. 먼저 `moreh-smi` 명령어를 이용해 현재 사용중인 MoAI Accelerator를 확인합니다. 

수행할 학습에 필요한 구체적인 MoAI Accelerator 설정에 대한 설명은 [3. 학습 실행하기](3_학습_실행하기.md)에서 제공하겠습니다.  

```bash
$ moreh-smi
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.2.0  Latest Version: 24.5.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  Large.256GB  |  -             |  -             |  -           |
+-------------------------------------------------------------------------------------------------+
```

