---
icon: terminal
tags: [tutorial, llama2]
order: 1000
---

# Llama3 70B Fine-tuning 

이 튜토리얼은 MoAI Platform에서 오픈 소스 [LLama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) 모델을 fine-tuning하는 예시를 소개합니다. 이 튜토리얼을 통해 MoAI Platform으로 AMD GPU 클러스터를 사용하는 방법을 익히고 향상된 성능과 자동 병렬화의 이점을 확인할 수 있습니다.


## 개요

MoAI Platform은 GPU를 손쉽게 제어할 수 있는 확장 가능한 AI 플랫폼으로, 수천 대의 GPU를 쉽게 제어하여 AI 모델을 학습하거나 추론할 수 있습니다. 모델을 파인튜닝하는 데 있어서 MoAI Platform의 특징은 가상화와 병렬화를 통해 고객에게 매우 간단한 학습 방법을 제안한다는 점입니다.

MoAI Platform은 여러 개의 GPU를 가상화하여 하나의 GPU인 MoAI Accelerator로 고객에게 제공합니다. 이는 학습 시 하나의 GPU만 사용하는 것처럼 보이게 하며, 다중 GPU를 사용하기 위해 필요한 사전 준비나 코드 작업이 필요하지 않습니다.

![](/overview/img_ov/v_3.png)


MoAI Platform은 고객이 가상화된 MoAI Accelerator를 사용할 때, 그 내부에서 자동으로 최적화된 병렬화를 제공합니다. 모델 크기와 데이터 크기에 따라 다양한 병렬화 방법을 고려하여 최적의 병렬화를 제공함으로써, 사용자가 추가적인 작업 없이도 간단한 코드로 매우 고성능의 학습을 경험할 수 있습니다.