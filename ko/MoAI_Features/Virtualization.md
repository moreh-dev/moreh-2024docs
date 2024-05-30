---
icon: cpu
order: 10
---

# GPU 가상화: MoAI Accelerator

MoAI Platform은 수십, 수백 GPU노드의 대형 GPU 클러스터를 MoAI Accelerator라는 단일 가속기로 가상화하여 사용자에게 제공합니다. 사용자는 멀티 노드 사용에 따른 모델 병렬화, 클러스터 환경 수동 설정 등의 작업에 대한 고민없이 단일 가속기를 사용하는 방식으로 모델을 설계하고 학습할 수 있습니다.

![](/overview/img_ov/v_3.png)


MoAI Accelerator는 터미널에 `moreh-smi` 명령어를 입력해 확인할 수 있습니다.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

출력된 내용을 확인해 보면 사용자는 2048 GB의 메모리를 갖춘 단일 가속기를 사용하는 것처럼 보입니다. 그러나 실제로는 각각 4개의 GPU로 구성된 4개의 노드를 사용하게 됩니다. 

가장 많이 사용되는 딥러닝 프레임워크 중 하나인 PyTorch에서 MoAI Accelerator를 제대로 인식하는지 확인해보겠습니다. Python 인터프리터에서 **`cuda`** API를 사용해 현재 사용 가능한 가속기의 수를 출력해보면 PyTorch가 MoAI Accelerator를 하나의 가속기로 인식하고 있음을 확인할 수 있습니다.

```bash
$ python
Python 3.8.19 (default) 
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.device_count()
1
```

여기서 중요한 점은 실제 사용자가 사용하는 환경에는 물리적인 GPU가 없다는 사실입니다. 사용자가 PyTorch와 같은 딥러닝 프레임워크에서 **`cuda`** 와 같은 API를 통해 GPU 가속기를 사용하려고 할 때, MoAI Platform은 GPU 클러스터 자원을 자동으로 할당합니다.

## MoAI Platform의 GPU 동적 할당

MoAI Platform은 GPU할당을 프로세스 단위로 동적으로 처리합니다. 때문에 사용자가 PyTorch와 같은 딥러닝 프레임워크를 사용하여 모델을 학습하고 추론할 때 물리GPU를 효율적으로 할당받을 수 있게 해줍니다. 이는 사용자가 미리 정의된 MoAI Accelerator의 다양한 flavor 중에서 원하는 GPU 수를 선택하고 필요에 따라 언제든지 조정할 수 있는 유연성을 제공합니다. 

반면에, 기존 클라우드 플랫폼들은 일반적으로 인스턴스를 생성하는 순간부터 물리적 GPU를 고정적으로 할당받습니다. 이는 사용자가 GPU 수를 변경하거나 더 이상 GPU를 사용하지 않으려 할 때, 기존 인스턴스를 삭제하거나 컨테이너를 종료한 후 재시작해야 하는 번거로움을 초래합니다. MoAI Platform의 동적 할당 방식은 이러한 불편을 획기적으로 줄여줍니다.

간단한 예시를 통해 MoAI Accelerator의 flavor를 변경하는 방법을 확인해보겠습니다.

먼저 터미널에 **`moreh-smi`** 명령어를 입력해 현재 사용중인 MoAI Accelerator를 확인합니다.

```bash
$ moreh-smi
+---------------------------------------------------------------------------------------------------+
|                                                  Current Version: 24.5.0  Latest Version: 24.5.0  |
+---------------------------------------------------------------------------------------------------+
|  Device  |        Name         |      Model     |  Memory Usage  |  Total Memory  |  Utilization  |
+===================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+---------------------------------------------------------------------------------------------------+
```


현재 사용자가 사용중인 MoAI Accelerator의 flavor는 [!badge variant="secondary" text=xLarge.512GB] 인 것을 확인할 수 있습니다. 더 큰 규모의 모델을 학습해야 하거나 학습 속도를 향상시키기 위해 더 많은 GPU를 사용하고 싶다면 간단하게  **`moreh-switch-model`** 명령어를 입력해 변경할 수 있습니다.

```bash
$ moreh-switch-model
Current MoAI Accelerator: xLarge.512GB

1. Small.64GB
2. Medium.128GB
3. Large.256GB
4. xLarge.512GB  *
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB
9. 6xLarge.3072GB
10. 8xLarge.4096GB
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB

Selection (1-13, q, Q): 8
The AI Accelerator model is successfully switched to  "4xLarge.2048GB".

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

Selection (1-13, q, Q):
q
```

다시 **`moreh-smi`** 명령어를 입력해 사용중인 MoAI Accelerator의 flavor를 확인해보면 [!badge variant="secondary" text=4xLarge.2048GB] 로 잘 변경된 것을 확인할 수 있습니다.

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

## 마무리

MoAI Platform은 MoAI Accelerator라는 가상화 기술을 통해 복잡한 멀티 노드 GPU 클러스터를 단순화하고, 사용자에게 강력하면서도 유연한 컴퓨팅 환경을 제공합니다. 사용자가 복잡한 설정과 관리 작업 없이도 모델 크기와 GPU 수를 자유롭게 조정할 수 있는 환경을 제공하여, 효율적인 리소스 활용이 가능하게 합니다. MoAI Platform에서 MoAI Accelerator를 활용해 보다 신속하고 효율적으로 딥러닝 모델을 설계하고 학습해 보세요.
