---
icon: note
tags: [guide]
order: 150
---

# MoAI Platform Features 

## **1. 가상 AI 가속기**

### GPU 동적 할당

- MoAI Platform이 제공하는 가상 AI 가속기의 동적 할당 기능으로 엔지니어는 필요한 만큼의 GPU 자원으로만 딥러닝 학습할 수 있습니다. 즉 연산이 실행되는 동안에만 GPU 자원이 할당되어 합리적인 비용으로 GPU 자원을 사용할 수 있습니다.
- 이로 인해 소프트웨어 및 인프라 개발 비용을 줄이는 데 도움이 되며, 개발 및 배포 시간을 단축할 수 있습니다.

### Pytorch/Tensorflow 호환성

- 딥러닝 모델의 발전에 따라 모델의 규모가 복잡해지고 연산량이 많아지면서 이를 학습하기 위한 파라미터도 수십억~수백억 단위의 규모로 증가합니다. 대규모 모델을 구현하려면 수많은 파라미터와 연산을 저장하기 위한 메모리 및 GPU 간 통신에 관련된 이슈를 해결하기 위한 작업을 해야 합니다. 하지만 이 작업은 극도로 많은 시간이 소요되며 하드웨어 구성 또는 모델 구조가 바뀔 때마다 처음부터 연산별로 최적의 구현 방식을 일일이 결정하는 것은 매우 번거로운 일입니다.
- MoAI Platform은 **자동 병렬화 기능**과 **대형 모델 학습을 위한 컴파일링 기능**을 제공하여 연산별로 최적의 구현방식을 자동으로 수행합니다.

## **2. 자동 병렬화**

- 인공지능 시대에는 LLM(대형 언어 모델) 및 LMM(대형 멀티모달 모델)과 같은 대규모 모델의 훈련 및 추론이 상당히 큰 GPU 클러스터와 효과적인 GPU 병렬화를 필요로 합니다.
- 현재 NVIDIA와 함께 사용되는 대부분의 AI 프레임워크는 모델의 크기와 복잡성, 그리고 사용 가능한 GPU 크기/클러스터에 따라 AI 엔지니어가 수동으로 병렬화해야 합니다. 이 설정 프로세스는 시간이 많이 소요되며 종종 몇 주가 걸립니다.
- MoAI 플랫폼은 특정 AI 모델과 GPU 클러스터의 크기를 기반으로 GPU 리소스를 최적으로 활용하는 Moreh AI 컴파일러를 통해 자동 병렬화를 제공합니다.
- 이는 NVIDIA와 같이 몇 주가 걸리는 AI 모델의 설정 및 배포 시간을 2~3일로 대폭 단축할 수 있습니다.
- **자동 병렬화는** MoAI Platform이 지원하는 핵심 기능 중 하나이며 모델 학습/추론 작업 시 자동으로 여러 가속기에 분배하여 병렬화합니다. 따라서 일반적으로 알려진 Data Parallelism, Model Parallelism, Pipeline Parallelism 같은 병렬화 전략들을 조합하여 최적의 데이터 연산 및 분배 방법을 알아서 결정하여 확장성을 극대화합니다.

### Advanced Parallelism

- MoAI Platform의 Advanced Parallelization은 기존 최적화 과정을 자동화함으로써, 최적의 병렬화 환경 변수 조합을 신속하게 결정합니다. 즉, 대규모 모델을 훈련하는 효율적인 Pipeline Parallelism, Tensor Parallelism의 최상의 매개변수와 환경변수 조합을 찾습니다.

## 3. 직관적이고 확장가능한 관리자 툴

- MoAI Platform이 제공하는 관리자 툴을 사용하면 고객별로 GPU 노드 사용 현황을 직관적으로 모니터링 하기 쉽습니다. 따라서 엔드유저가 사용한 GPU 자원 및 권한을 직접 관리할 수 있습니다.
- 애플리케이션의 규모가 커짐에 따라 더 많은 GPU 자원을 효과적으로 관리할 수 있는 구조를 갖췄으며 발생 가능한 GPU 부족 및 온도 문제에 대해 정확한 원인을 파악하고 신속하게 대응하고 해결할 수 있습니다.

## MoAI Platform 적용시 인프라 구조

### 학습 실행 흐름

![](./img/features_01.png)

1. Front Node 의 VM 상에서 `python train.py` 와 같이 프로세스를 실행시키게 되면, VM에 설치되어 있는 Moreh 솔루션 전용 Pytorch가 Back Node 의 GPU로 연결될 수 있도록 SDAManager에게 요청이 가게 합니다.
2. Master Node 에는 Moreh 솔루션의 관리 모듈인 SDAManager 에서 해당 통신요청을 받게 되고, 요청 (GPU 갯수 등) 을 분석하여 비어있는 Back Node에 요청을 할당하도록 합니다.
3. Worker Agent 는 요청을 받아서 실제로 GPU 연산을 수행하는 Job을 각 노드들에 실행하게 됩니다. 실행된 Job은 Worker process 를 통해 실행하게 됩니다.
4. 실행된 Worker 는 VM 의 Python process 와 InfiniBand 망을 통해서 통신을 맺게 되고 학습이 수행됩니다.


## MoAI Platform 아키텍처

![](./img/features_02.png)

### 노드 구성

크게 3가지의 노드 그룹으로 구성되어 있습니다. 

1. **Master Node**: Front 와 Back 의 통신을 중개하고 Moreh 솔루션 사용 현황을 관리합니다. Front Node에서 Moreh 솔루션에 학습 요청이 있을 경우, 최초로 Master Node에서 요청을 받고 Back Node와의 통신을 중개하게 됩니다. Master Node 의 S/W 및 H/W 스택은 모두 모레에서 관리, 운영하고 있습니다. 

2. **Front Node**: 유저가 접속하여 사용할 수 있는 VM을 제공하기 위한 노드그룹으로, Openstack 으로 구성하여 VM을 생성하고 있습니다. Openstack 등 Front Nodes 에 올라간 S/W stack은 락플레이스에서 운영 및 관리하고 있으며, H/W는 모레에서 운영 및 관리하고 있습니다. 

3. **Back Node**: 실제 GPU 연산이 수행되는 노드 그룹으로 각 노드에는 AMD MI250 GPU 4장으로 구성되어 있습니다. Back Node의 S/W 및 H/W 스택은 모두 모레에서 관리, 운영하고 있습니다.

### 서비스 구성

- **Front Node**의 VM 상에서는 다음과 같은 서비스를 제공합니다.
    - **Moreh 솔루션 전용 Pytorch**
        - 현재 Pytorch는 `1.13.1`, `1.10.0`, `1.7.1` 버전을, Tensorflow는 `2.9` 버전을 제공하고 있습니다. Moreh 솔루션 전용 Pytorch/Tensorflow는 공개 패키지와 내용이 거의 동일하나, GPU 연산을 수행하게 되면, 해당 VM 내에 GPU 디바이스를 찾는 것이 아닌, Back Node 의 GPU에서 디바이스를 할당받고 연산이 수행될 수 있도록 일부 코드를 수정한 버전입니다. 기능 및 성능 면에서는 공개 패키지와 동일하다고 보시면 됩니다.
    - **moreh-toolkit**
        - `moreh-smi`: GPU 연산 시 메모리 사용량, 프로세스 현황 등을 확인할 수 있습니다. (NVIDIA의 *nvidia-smi* 와 비슷한 툴입니다.)
        - `moreh-switch-model`: AI 가속기를 변경할 수 있습니다. GPU의 사이즈를 변경하고 싶을 때 사용하며, 미리 원하는 AI 가속기로 변경을 하고 프로세스를 실행하여야, 해당 프로세스가 변경된 AI 가속기, 즉 변경된 GPU 사이즈에서 실행됩니다.
        - `update-moreh`: Moreh 솔루션을 최신버전을 업데이트 하거나 롤백할 수 있습니다.
        - `moreh-docker-run`: HAC 서비스에서는 Moreh 솔루션이 설치된 도커 이미지를 제공하고 있습니다. 컨테이너 내에서 학습을 수행하고 싶은 유저는 이 툴을 활용하여 쉽게 모레 도커 이미지를 다운받아 실행할 수 있습니다.
        - `get-reference-model`: HAC 서비스에서는 Moreh 솔루션 상에서 동작이 검증된 Pytorch 및 Tensorflow의 Reference Model 을 완성된 코드레벨로 제공하고 있습니다. 유저는 이 툴을 활용하여 원하는 Reference Model 모델을 쉽게 VM 내에 다운받아 학습을 수행할 수 있습니다. 어떤 모델이 제공되는지는 HAC 공식 홈페이지의 [Hyperscale AI Computing 모델 구성](https://cloud.kt.com/solution/hyperscaleAiComputing/?tab=0) 를 참고 바랍니다.
- **Master Node**의 SDAManager는 Kubernetes 환경에서 동작하며, 현재 3대의 물리서버로 고가용성을 보장하고 있습니다.
    - `Core API`: SDAManager 의 핵심 모듈들이 들어있는 서비스입니다. SDAManager의 모든 기능들은 본 서비스를 통해 거쳐간다고 보시면 됩니다. Core 모듈을 호출할 수 있는 API들이 gRPC 프로토콜을 통해 제공되고 있습니다. Front Node, Back Node 를 제외한 외부에서의 호출은 불가능하도록 보안정책을 적용하고 있습니다.
    - `Schduler`: Front Node에서 학습 요청이 있을 경우, 어떤 Back Node 로 얼마나 많은 자원을 할당해야 할지를 관리하는 스케줄러 서비스입니다.
    - `DB`: SDAManager 관리에 필요한 메타 데이터, 유저 데이터, GPU 사용관련 데이터 등 SDAManager 와 관련된 모든 데이터를 저장하고 있습니다.
- **Back Node**는 실제 GPU 연산이 수행되는 노드들이며, 아래와 같은 2개의 서비스로 구성되어 있습니다.
    - `Worker Agent`: SDAManager로부터 GPU 할당 및 해제 요청을 받을 경우 해당 Back Node의 Worker 프로세스를 실행 및 중지 등을 관리합니다. 또한 해당 노드의 S/W 또는 H/W 장애를 감지하는 역할을 하기도 합니다.
    - `Worker`: GPU 연산을 수행하는 프로세스입니다. 요청이 없을 경우에는 프로세스가 실행되어 있지 않다가, Worker Agent가 요청을 받을 경우 이 Worker 프로세스를 실행합니다. 학습이 종료되면 Worker 프로세스도 함께 종료됩니다.