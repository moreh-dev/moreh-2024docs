---
icon: terminal
tags: [guide]
order: 90
expanded: false
---

# Advanced Parallelization(AP)

### What is Advanced Parallelization?

Advanced Parallelization (AP) is an optimized automated distributed parallel processing feature provided by the MoAI Platform. Typically, ML engineers go through numerous trial-and-error processes to optimize model parallelization when training large models. This involves considering the memory size of the GPUs being used, directly applying various parallelization techniques, and measuring the performance of different combinations of options within each technique to determine the optimal configuration.

However, with the AP feature offered by the MoAI Platform, you can easily apply optimized parallelization techniques, significantly reducing the time and effort required before starting the training process.


![](overview_05.png)

Moreh의 AP 기능은 기존 최적화 과정을 **자동화**함으로써, 최적의 병렬화 환경 변수 조합을 신속하게 결정합니다. 따라서 대규모 모델 훈련시 적용하는 효율적인 Pipeline Parallelism, Tensor Parallelism의 최적 매개변수와 환경 변수 조합을 간단히 얻을 수 있습니다.
