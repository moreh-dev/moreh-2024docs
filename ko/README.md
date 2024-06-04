---
icon: book
order: 2000
expanded: true
outbound:
  enabled: false
---

# MOREH DOCS

**MoAI(Moreh AI appliance for AI accelerators)** 는 대규모 딥러닝 모델 개발에 필수적인 그래픽 처리 장치(GPU)를 손쉽게 제어할 수 있는 확장 가능한 AI 플랫폼입니다.

----

### Getting Started

   | 
---    | ---
 [**Fine-tuning 시작하기**](/Tutorials/index.md) <br> MoAI Platform을 처음 사용하는 사용자에게 필요한 정보 안내 | [ **AP 가이드**](/Supported_Documents/AP/index.md) <br> Advanced Parallelization (AP) 기능 사용 안내 
[ **Moreh Toolkit**](/Supported_Documents/moreh_toolkit.md) <br> command line 사용방법 |[ **MoAI Platform Features**](/MoAI_Features/index.md) <br> MoAI Platform의 가상화와 병렬화 기능

!!!primary Note
MoAI Platform은 현재 개발이 계속 진행되고 있습니다. 따라서, 문서의 내용은 언제든지 변경될 수 있습니다.
!!!

## MoAI Platform 핵심 기술

![](./img/overview_01.png)

딥러닝 모델이 발전하면서 수십억, 수백억 개의 파라미터를 포함하는 복잡한 구조가 되었고, 이에 따라 대규모 컴퓨팅 자원이 AI  인프라의 중요한 부분이 되었습니다. 대규모 컴퓨팅 자원을 사용하여 모델을 개발하려면 모델의 병렬 처리와 클러스터 환경의 수동 설정과 같이 학습 프로세스를 최적화하는 과정이 필수적입니다. 특히, GPU 및 노드 관리를 통한 학습 최적화는 개발자들에게 많은 시간과 노력을 요구합니다.

MoAI Platform은 이러한 문제를 해결하기 위해 다음과 같은 기능을 제공하여 대규모 AI 시대에 효율적인 인프라를 지원합니다.

1. **[다양한 가속기, 다중 GPU 지원](https://docs.moreh.io/ko/overview/#1-%EB%8B%A4%EC%96%91%ED%95%9C-%EA%B0%80%EC%86%8D%EA%B8%B0-%EB%8B%A4%EC%A4%91-gpu-%EC%A7%80%EC%9B%90)**
2. **[GPU 가상화](https://docs.moreh.io/ko/overview/#2-gpu-%EA%B0%80%EC%83%81%ED%99%94)**
3. **[동적 GPU 할당](https://docs.moreh.io/ko/overview/#3-%EB%8F%99%EC%A0%81-gpu-%ED%95%A0%EB%8B%B9)**
4. **[AI Compiler 자동 병렬화](https://docs.moreh.io/ko/overview/#4-ai-compiler-%EC%9E%90%EB%8F%99-%EB%B3%91%EB%A0%AC%ED%99%94)**
