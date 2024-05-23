---
icon: note
tags: [guide]
order: 5
---

# MoAI Platform 병렬화

MoAI Platform에서 사용자는 가상의 하나의 GPU만 사용하게 됩니다. 따라서 사용자는 하나의 GPU를 사용하는 코드를 작성하게 됩니다. 그렇다면 **어떻게 여러 개의 GPU를 사용할 수 있을까요?**

사용자가 여러 GPU를 사용하는 flavor를 선택하면, MoAI Platform은 자동으로 데이터 병렬화를 수행합니다. 예를 들어, 사용자가 8개의 device를 포함한 flavor를 선택하면, 전체 배치 크기(batch size)를 8등분하여 각 device에 나눠 처리하고, 이를 통해 학습 속도가 크게 향상됩니다.

예를 들어, 사용자가 llama3-8b 모델을 fine-tuning할 때 4개의 GPU를 사용하는 flavor를 선택하고, batch size를 16로 설정한다면 각 GPU당 4개로 아래와 같은 throughput이 나올 것입니다.

```bash
# Llama3-8b-base fine-tuning, batch-size 16, gpu 4
[Step 4/17944] | Loss: 2.03125 | Duration: 1.27 | Throughput: 12882.87 tokens/sec
[Step 6/17944] | Loss: 2.03125 | Duration: 1.22 | Throughput: 13393.38 tokens/sec
[Step 8/17944] | Loss: 2.109375 | Duration: 1.31 | Throughput: 12492.66 tokens/sec
[Step 10/17944] | Loss: 2.015625 | Duration: 1.24 | Throughput: 13201.98 tokens/sec
```

또한, 사용자가 16개의 GPU를 사용하는 flavor를 선택하고 batch size를 64로 설정하면, 각 GPU당 동일하게 4개로 자동으로 병렬처리가 되어 throughput은 4개의 GPU를 사용할 때보다 약 4배가 될 것 입니다.

```bash
# Llama3-8b-base fine-tuning, batch-size 64, gpu 16
[Step 4/4486] | Loss: 2.125 | Duration: 1.42 | Throughput: 46148.86 tokens/sec
[Step 6/4486] | Loss: 2.078125 | Duration: 1.33 | Throughput: 49221.88 tokens/sec
[Step 8/4486] | Loss: 2.03125 | Duration: 1.33 | Throughput: 49392.99 tokens/sec
[Step 10/4486] | Loss: 2.046875 | Duration: 1.24 | Throughput: 52744.78 tokens/sec
```

만약 사용자가 **더 큰 메모리를 사용하기 위해 여러 GPU를 사용하려는 경우**는 어떨까요?

MoAI Platform은 모델 병렬화와 최적화를 자동으로 지원합니다.

사용자가 Llama3-8b 모델을  16개의 GPU로, batch size를 512로 설정하여 돌린다면, 모델 병렬화와 데이터 병렬화가 동시에 이루어져 학습이 진행됩니다.

```bash
## This snippet is fake
# Llama3-8b-base fine-tuning, batch-size 512, gpu 16
[Step 4/4486] | Loss: 2.125 | Duration: 1.42 | Throughput: 46148.86 tokens/sec
[Step 6/4486] | Loss: 2.078125 | Duration: 1.33 | Throughput: 49221.88 tokens/sec
[Step 8/4486] | Loss: 2.03125 | Duration: 1.33 | Throughput: 49392.99 tokens/sec
[Step 10/4486] | Loss: 2.046875 | Duration: 1.24 | Throughput: 52744.78 tokens/sec
```

그 외에도  70B와 같은 대형 모델도 자동으로 병렬화되어 간편하게 학습할 수 있습니다. 이처럼 MoAI Platform은 사용자가 사용하는 모델과 배치 크기 등에 따라 자동으로 **최적화와 병렬화를 제공하여, 다중 GPU를 편리하고 효율적으로 사용할 수 있도록 합니다.**