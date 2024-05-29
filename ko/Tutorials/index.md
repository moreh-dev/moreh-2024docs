---
icon: terminal
tags: [tutorial]
expanded: true
order: 80
---

# Fine-tuning 튜토리얼

이 튜토리얼은 Llama2, Mistral 등의 대형 언어 모델을 fine-tuning 하고자 하는 모든 분들을 위한 것입니다. MoAI 플랫폼을 사용하여 아래 대형 언어 모델들을 미세 조정하는 과정을 안내합니다.


- [Llama3 8B](../Tutorials/llama3_8b_tutorial/index.md)
- [Llama3 70B](../Tutorials/llama3_70b_tutorial/index.md)
- [Mistral](../Tutorials/mistral_tutorial/index.md)
- [GPT](../Tutorials/gpt_tutorial/index.md)
- [Qwen](../Tutorials/qwen_Tutorial/index.md)
- [Baichuan2](../Tutorials/baichuan2_tutorial/index.md)
- [Llama2 13B](../Tutorials/llama2_13b_tutorial/index.md)

머신러닝에서 미세 조정(fine-tuning)이란 사전 학습된 모델의 매개변수를 새로운 데이터로 조정하여 특정 작업의 성능을 향상시키는 것을 의미합니다. 즉, 기존 모델을 새로운 작업에 적용하고자 할 때, 새로운 데이터셋으로 모델을 최적화하여 특정 요구와 도메인에 맞게 커스터마이징하는 과정입니다.

사전학습된 모델은 범용성을 고려한 매우 큰 파라미터를 가지는 모델이며 큰 모델을 효과적으로 fine-tuning하려면 충분한 양의 학습 데이터가 필요합니다.

MoAI Platform에서는 GPU의 메모리 사이즈를 고려해 최적화된 병렬화 기법을 손쉽게 적용할 수 있어 학습 시작 전에 소요되는 시간과 노력을 획기적으로 줄일 수 있습니다. 


#### 이 튜토리얼에서 배우게 될 내용:

1. 데이터셋, 모델, 토크나이저 로드하기
2. 학습 실행 및 결과 확인하기
3. 자동 병렬화 기능 적용하기
4. 적절한 학습 환경 및 AI 가속기 선택 방법
