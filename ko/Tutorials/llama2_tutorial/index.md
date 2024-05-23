---
icon: terminal
tags: [guide]
order: 1000
---

# Llama2 Fine-tuning

이 튜토리얼은 MoAI Platform에서 오픈 소스 [Llama2 13B](https://huggingface.co/meta-llama/Llama-2-13b-hf)을 fine-tuning하는 예시를 소개합니다. 튜토리얼을 통해 MoAI Platform을 통해 AMD GPU 클러스터를 활용하는 방법을 배우고, 성능 및 자동 병렬화의 장점을 확인할 수 있습니다.

### 개요

Llama2 모델은 2023년 7월에 [Meta](https://about.meta.com/)가 공개한 Decoder-only Transformer 기반 오픈 소스 모델입니다. 기존 Llama 모델의 구조를 따르지만 40% 더 많은 데이터로 학습시켜 더 다양하고 복잡한 정보를 이해할 수 있습니다.

Llama2는 특히 언어 이해 및 생성 작업에 있어서 뛰어난 성능을 보이며, 다양한 자연어 처리 태스크에서 SOTA 성능을 달성하였습니다. 이 모델은 다국어 지원이 가능하여 전 세계 다양한 언어의 텍스트를 처리할 수 있으며, 공개적으로 접근 가능하여 연구 및 개발 목적으로 널리 사용될 수 있습니다.

이 튜토리얼에서는 MoAI Platform에서 [CNN Daily Mail](https://huggingface.co/datasets/cnn_dailymail) 데이터셋을 활용하여 Llama2 모델을 fine-tuning하는 방법을 살펴보겠습니다.  이 과정에서는 자연어 처리 방법 중 하나인 요약(summarize) 작업을 다룰 것입니다.


### 시작하기 전에

MoAI Platform 상의 컨테이너 혹은 가상 머신을 인프라 제공자로부터 발급받고, 여기에 SSH로 접속하는 방법을 안내 받으시기 바랍니다. 예를 들어 MoAI Platform 기반으로 운영되는 다음 퍼블릭 클라우드 서비스를 신청하여 사용할 수 있습니다.

[!ref target="blank" text="KT Cloud의 Hyperscale AI Computing"](https://cloud.kt.com/solution/hyperscaleAiComputing/)

혹은 일시적으로 체험판 컨테이너 및 GPU 자원을 할당 받기를 원하시는 분은 Moreh(support@moreh.io)에 문의하시기 바랍니다.

SSH로 접속한 다음 `moreh-smi` 명령을 실행하여 MoAI Accelerator가 잘 표시되는지 확인하시기 바랍니다. 디바이스 이름은 시스템마다 다르게 설정되어 있을 수 있습니다. 

#### MoAI Accelerator 확인

이 튜토리얼에서 안내할 Llama2 모델과 같은 sLLM을 학습하기 위해서는 적절한 크기의 MoAI Accelerator를 선택해야 합니다. 먼저 `moreh-smi` 명령어를 이용해 현재 사용중인 MoAI Accelerator를 확인합니다. 

수행할 학습에 필요한 구체적인 MoAI Accelerator 설정에 대한 설명은 [3. 학습 실행하기](3_학습_실행하기.md) 에서 제공하겠습니다. 

```bash
$ moreh-smi
23:44:25 April 18, 2024
+---------------------------------------------------------------------------------------------------+
|                                                  Current Version: 24.2.0  Latest Version: 24.3.0  |
+---------------------------------------------------------------------------------------------------+
|  Device  |        Name         |      Model     |  Memory Usage  |  Total Memory  |  Utilization  |
+===================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+---------------------------------------------------------------------------------------------------+
```

