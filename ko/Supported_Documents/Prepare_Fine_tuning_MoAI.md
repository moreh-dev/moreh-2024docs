---
icon: terminal
tags: [guide]
order: 30
breadcrumb: true
---

# MoAI Platform에서 Fine-tuning 준비하기

MoAI Platform은 다양한 GPU로 구성될 수 있지만, 동일한 인터페이스(CLI)를 통해 사용자에게 일관된 경험을 제공합니다. 모든 사용자가 같은 방식으로 시스템에 접근하여 플랫폼을 사용할 수 있기 때문에 보다 효율적이며 직관적입니다.

MoAI Platform 또한 일반적인 AI 학습 환경과 유사하게  Python 기반의 프로그래밍을 지원합니다. 이에 따라 본 문서에서는 AI 학습을 위한 표준 환경 구성으로서 conda 가상 환경의 설정과 사용 방법을 중심으로 설명합니다.

## conda 환경 설정하기
1. 훈련을 시작하기 위해 먼저 conda 환경을 생성합니다.
    
    ```bash
    $ conda create --name <my-env> python=3.8
    ```
    
    `<my-env>` 에는 사용자가 사용할 환경 이름을 입력합니다.
    
2. conda 환경을 활성화합니다.
    
    ```bash
    $ conda activate <my-env>
    ```
    
3. MoAI Platform은 다양한 PyTorch 버전을 제공하고 있으므로 사용자가 필요한 환경에 맞는 버전을 선택해 설치할 수 있습니다. 
    
    ```bash
    $ pip install torch==1.13.1+cu116.moreh24.5.0
    ```

4. `moreh-smi` 명령어를 입력해 설치된 MoAI Platform의 버전과 사용중인 [MoAI Accelerator](/MoAI_Features/Virtualization.md) 정보를 확인할 수 있습니다. 현재 사용중인 MoAI Accelerator는 [!badge variant="secondary" text=4xLarge.2048GB] 입니다.
    
    ```bash
    $ moreh-smi
    +-----------------------------------------------------------------------------------------------------+
    |                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
    +-----------------------------------------------------------------------------------------------------+
    |  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
    +=====================================================================================================+
    |  * 0     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
    +-----------------------------------------------------------------------------------------------------+
    ```


!!! 
각 모델별로 MoAI Platform에서 권장하는 Fine-tuning 시 최적의 파라미터는 [LLM Fine-tuning 파라미터 가이드](LLM_param_guide.md) 를 참고하시기 바랍니다.
!!!


!!! 
`moreh-smi` , `moreh-switch-model` 를 비롯한 moreh toolkit의 구체적인 사용 방법에 대해서는 [MoAI Platform의 toolkit 사용하기](moreh_toolkit.md) 를 참고하시기 바랍니다.
!!!



