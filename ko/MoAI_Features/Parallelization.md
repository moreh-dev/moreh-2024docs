---
icon: note
order: 5
---

# 자동 병렬화

MoAI Platform에서 사용자는 가상화된 하나의 GPU인 MoAI Accelerator를 사용하게 됩니다. 따라서 MoAI Platform 상에서 사용자는 별도 병렬화 작업없이 하나의 GPU를 사용하는 것을 가정하고 코드를 작성하게 됩니다. 그렇다면 MoAI Platform에서는 어떻게 여러 개의 GPU를 사용할 수 있을까요?

**MoAI Platform은 사용하는 GPU 수에 따라 자동으로 가장 좋은 최적화 및 병렬화를 수행합니다.** 

예를 들어, 사용자가 8개의 GPU를 사용하는 가속기 flavor를 선택하여 학습을 시작하면 MoAI Platform은 자동으로 전체 배치 크기를 8등분하여 각 GPU에 분배하여 처리합니다. 더 자세한 설명을 위해 Llama3-8b 모델을 fine-tuning하는 상황을 가정해보겠습니다. 사용자가 4개의 GPU를 사용하는 가속기 flavor를 선택하고 배치 크기를 16으로 설정한다면, 각 GPU당 4개의 배치가 자동으로 분배되어 대략 1초에 12만 5천 개의 토큰을 처리하는 것을 확인할 수 있습니다.

```bash
# Llama3-8b-base fine-tuning, batch-size 16, gpu 4
[Step 4/17944] | Loss: 2.03125 | Duration: 1.27 | Throughput: 12882.87 tokens/sec
[Step 6/17944] | Loss: 2.03125 | Duration: 1.22 | Throughput: 13393.38 tokens/sec
[Step 8/17944] | Loss: 2.109375 | Duration: 1.31 | Throughput: 12492.66 tokens/sec
[Step 10/17944] | Loss: 2.015625 | Duration: 1.24 | Throughput: 13201.98 tokens/sec
```

더 빠르고 효율적인 학습을 위해 더 많은 GPU를 사용하는 가속기 flavor를 선택하고 배치 크기를 키울 수도 있습니다. 사용자가 16개의 GPU를 사용하는 가속기 flavor를 선택하고 배치 크기를 64로 설정한 후 학습을 시작하면, 1초에 처리되는 토큰 수를 기존 대비 4배로 키울 수 있습니다.

```bash
# Llama3-8b-base fine-tuning, batch-size 64, gpu 16
[Step 4/4486] | Loss: 2.125 | Duration: 1.42 | Throughput: 46148.86 tokens/sec
[Step 6/4486] | Loss: 2.078125 | Duration: 1.33 | Throughput: 49221.88 tokens/sec
[Step 8/4486] | Loss: 2.03125 | Duration: 1.33 | Throughput: 49392.99 tokens/sec
[Step 10/4486] | Loss: 2.046875 | Duration: 1.24 | Throughput: 52744.78 tokens/sec
```

그런데 만약 사용자가 보다 더 **큰 배치 크기를 사용하려는 경우**는 어떨까요? 추가적인 코드 수정이 없다면, 일반적인 GPU 클러스터에서는 Out of Memory(OOM) 에러가 발생할 가능성이 큽니다. 하지만 MoAI Platform은 자동으로 모델을 병렬화하여 학습을 진행시킬 수 있습니다.

사용자가 16개의 GPU를 사용하는 가속기 flavor를 선택하고, 배치 크기를 512로 설정하여 학습하면 자동으로 모델 병렬화와 데이터 병렬화가 동시에 적용돼 같은 개수의 GPU를 사용하더라도 더 큰 배치 크기로 학습할 수 있습니다.

```bash

# Llama3-8b-base fine-tuning, batch-size 512, gpu 16
[Step 4/560] | Loss: 1.953125 | Duration: 24.00 | Throughput: 21844.08 tokens/sec
[Step 6/560] | Loss: 1.8671875 | Duration: 24.63 | Throughput: 21283.67 tokens/sec
[Step 8/560] | Loss: 2.0 | Duration: 24.41 | Throughput: 21475.45 tokens/sec
[Step 10/560] | Loss: 1.9609375 | Duration: 24.26 | Throughput: 21609.36 tokens/sec
[Step 12/560] | Loss: 1.90625 | Duration: 24.43 | Throughput: 21463.95 tokens/sec
```

이 외에도 70B와 같은 대형 모델도 별도의 추가 작업없이 자동으로 병렬화되기 때문에 간편하게 학습할 수 있습니다. 이처럼 MoAI Platform은 사용자가 사용하는 모델과 배치 크기 등에 따라 자동으로 **최적화와 병렬화를 제공하여, 다중 GPU를 편리하고 효율적으로 사용할 수 있도록 합니다.**
