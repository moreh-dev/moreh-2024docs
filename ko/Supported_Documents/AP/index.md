---
icon: terminal
tags: [guide]
order: 80
---

# Advanced Parallelization (AP)

## MoAI Platform의 자동병렬화 기능,  Advanced Parallelization (AP)

### 병렬화가 반드시 필요한 이유는 무엇일까요?

일반적으로 많이 사용되는 Llama2 13B 모델의 크기를 Bytes 단위로 계산해봅시다.

130억개의 파라미터를 가진 Llama2 13B는 FP16 데이터 형을 기준으로 13B * 2 bytes 크기입니다. 이는 약 24.2GB입니다. AdamW optimizer는 13B * 2 * 2로 약 48.4GB입니다. 모델을 로드하는 데에만 최소 72.6GB가 필요하며 Gradient에 필요한 메모리 약 24.2GB 등 모델 로드 이외에 +$\alpha$ 의 메모리가 필요합니다. 

MoAI Platform은 가상화 된 GPU 하나를 제공하지만, 이는 실제로 여러 GPU에 모델을 복사하고 batch samples를 균등하게 분할하여 학습시키는 DDP 방식으로 기본 설정되어 동작합니다. 따라서 1 device chip의 VRAM인 64GB가 넘는 데이터를 로드할 수 없습니다.

따라서 모델을 병렬화하여 여러 GPU에 로드하는 기법이 필요합니다.

### Advanced Parallelization이란?

`Advanced Parallelization`(이하 `AP`)은 MoAI Platform에서 제공하는 **최적화된 분산 병렬처리** 기능입니다. 일반적으로 ML 엔지니어라면 모델 병렬화를 ‘최적화’하기 위해 수 많은 경험적 시행착오를 겪곤합니다. (예를 들어, 모델의 stages 개수, micro batches 개수 등) 하지만 MoAI Platform을 사용한다면 다른 프레임워크에서는 경험할 수 없는 특별한 AP 기능을 활용할 수 있어 최적화된 병렬화에 소요되는 시간과 노력을 획기적으로 줄일 수 있습니다.

Moreh의 AP 기능은 기존 최적화 과정을 **자동화**함으로써, 최적의 병렬화 환경 변수 조합을 신속하게 결정합니다. 따라서 대규모 모델 훈련시 적용하는 효율적인 Pipeline Parallelism, Tensor Parallelism의 최적 매개변수와 환경 변수 조합을 간단히 얻을 수 있습니다.
