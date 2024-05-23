---
icon: terminal
tags: [guide]
order: 600
---

# Baichuan2 Fine-tuning

이 튜토리얼은 MoAI Platform에서 오픈 소스 [Baichuan2 13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base) 모델을 fine-tuning하는 예시를 소개합니다. 튜토리얼을 통해 MoAI Platform으로 AMD GPU 클러스터를 사용하는 방법을 익히고 성능 및 자동 병렬화의 이점을 확인할 수 있습니다.


## 개요

Baichuan2는 [Baichuan Intelligent Technology](https://github.com/baichuan-inc)가 개발한 오픈 소스, 대규모 다국어 언어 모델입니다. 이 모델은 2조 6천억 개의 토큰으로 구성된 방대한 데이터 세트에서 훈련된 70억 및 130억 매개 변수로 사용할 수 있는 구성을 가지고 있습니다. 

이 튜토리얼에서는 MoAI Platform에서 text-generation e-commerce 데이터셋인 [Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)을 활용해  Baichuan2 13B 모델을 fine-tuning 해보겠습니다.

## 시작하기 전에

MoAI Platform 상의 컨테이너 혹은 가상 머신을 인프라 제공자로부터 발급받고, 여기에 SSH로 접속하는 방법을 안내 받으시기 바랍니다. 예를 들어 MoAI Platform 기반으로 운영되는 다음 퍼블릭 클라우드 서비스를 신청하여 사용할 수 있습니다.

- KT Cloud의 Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

혹은 일시적으로 체험판 컨테이너 및 GPU 자원을 할당 받기를 원하시는 분은 Moreh(support@moreh.io)에 문의하시기 바랍니다.

SSH로 접속한 다음 `moreh-smi` 명령을 실행하여 MoAI Accelerator가 잘 표시되는지 확인하시기 바랍니다. 디바이스 이름은 시스템마다 다르게 설정되어 있을 수 있습니다.

### MoAI Accelerator 확인

이 튜토리얼에서 안내할 Mistral 7B 모델과 같은 sLLM을 학습하기 위해서는 적절한 크기의 MoAI Accelerator를 선택해야 합니다. 먼저 `moreh-smi` 명령어를 이용해 현재 사용중인 MoAI Accelerator를 확인합니다. 

수행할 학습에 필요한 구체적인 MoAI Accelerator 설정에 대한 설명은 [3. 학습 실행하기](3_학습_실행하기.md) 에서 제공하겠습니다.  



```bash
$ moreh-smi
11:40:36 April 16, 2024
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.3.0  Latest Version: 24.3.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  Large.256GB  |  -             |  -             |  -            |
+-------------------------------------------------------------------------------------------------+
```

