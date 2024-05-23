---
icon: terminal
tags: [guide]
order: 90
---

# Prepare Fine-tuning on MoAI Platform

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
    
3. Fine-tuning에 필요한 library와 package를 설치합니다.
    
    ```bash
    $ pip install torch==1.13.1+cu116.moreh24.3.0
    $ pip install transformers==4.34.0
    $ pip install datasets==2.14.5
    $ pip install loguru==0.7.2
    ```
    
4. `moreh-smi` 명령어를 입력해 설치된 Moreh 솔루션의 버전과 사용중인 MoAI Accelerator 정보를 확인할 수 있습니다. 현재 사용중인 MoAI Accelerator는 `4xLarge.2048GB`입니다. MoAI Accelerator에 대한 자세한 정보는 MoAI Accelerator 사양을 참고해주세요.
    
    ```bash
    $ moreh-smi
    11:07:03 April 01, 2024
    +-----------------------------------------------------------------------------------------------------+
    |                                                    Current Version: 24.3.0  Latest Version: 24.3.0  |
    +-----------------------------------------------------------------------------------------------------+
    |  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
    +=====================================================================================================+
    |  * 0     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
    +-----------------------------------------------------------------------------------------------------+
    ```
    

# MoAI Accelerator 선택, 변경하기

sLLM 파인튜닝시 학습 데이터 배치 사이즈에 따른 적절한 MoAI Accelerator 모델을 moreh toolkit을 사용하여 선택, 변경할 수 있습니다. 참고로, sLLM(약 7B~13B 모델)을 fine-tuning 하기 위해 일반적으로 사용되는 데이터셋의 크기는 약 40GB의 텍스트 데이터셋입니다. 

먼저, `moreh-smi` 를 사용하여 현재 사용하고 있는 MoAI Accelerator 모델을 확인해 보겠습니다. 

```bash
$ moreh-smi
12:01:30 April 12, 2024
+------------------------------------------------------------------------------------------------+
|                                                     Current Version:   Latest Version: 24.2.0  |
+------------------------------------------------------------------------------------------------+
|  Device  |    Name            |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+================================================================================================+
|  * 0     |  MoAI Accelerator  |  Small.64GB  |  -             |  -             |  -            |
+------------------------------------------------------------------------------------------------+
```

현재  사용하고 있는 MoAI Accelerator 모델에서 제공되는 메모리는 64GB입니다.  `moreh-switch-model` 을 사용하여 더 큰 메모리를 제공하는 MoAI Accelerator 모델로 변경해 보겠습니다. 

```bash
$ moreh-switch-model
Current MoAI Accelerator: Small.64GB

1. Small.64GB  *
2. Medium.128GB
3. Large.256GB
4. xLarge.512GB
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB
9. 6xLarge.3072GB
10. 8xLarge.4096GB
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB

Selection (1-13, q, Q):
```

`4xLarge.2048GB` 모델로 변경하기 위해 `8`을 입력합니다. 

```bash
Selection (1-13, q, Q): 8
The MoAI Accelerator model is successfully switched to  "4xLarge.2048GB".

1. Small.64GB
2. Medium.128GB
3. Large.256GB
4. xLarge.512GB
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB  *
9. 6xLarge.3072GB
10. 8xLarge.4096GB
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB

Selection (1-13, q, Q): q
```

 `q` 를 입력하여 변경을 완료합니다. 

다시 `moreh-smi` 를 사용하여 변경된 상태를 확인하면 사용하고 있는 MoAI Accelerator 모델이 `4xLarge.2048GB` 모델로 변경된 것을 확인할 수 있습니다. 

```bash
$ moreh-smi
12:02:19 April 12, 2024
+----------------------------------------------------------------------------------------------------+
|                                                         Current Version:   Latest Version: 24.2.0  |
+----------------------------------------------------------------------------------------------------+
|  Device  |    Name            |     Model        |  Memory Usage  |  Total Memory  |  Utilization  |
+====================================================================================================+
|  * 0     |  MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
+----------------------------------------------------------------------------------------------------+
```


!!! 
각 모델별로 MoAI Platform에서 권장하는 Fine-tuning 시 최적의 파라미터는 [LLM Fine-tuning 파라미터 가이드](LLM%20Fine-tuning%20파라미터%20가이드.md) 를 참고하시기 바랍니다.
!!!


!!! 
`moreh-smi` , `moreh-switch-model` 를 비롯한 moreh toolkit의 구체적인 사용 방법에 대해서는 [MoAI Platform의 toolkit 사용하기](//) 를 참고하시기 바랍니다.
!!!



