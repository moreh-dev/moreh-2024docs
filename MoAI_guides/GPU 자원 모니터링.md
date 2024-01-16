---
order: 80
icon: terminal
tags: [guide]
---
# GPU 자원 모니터링 (moreh-smi)

**AI 가속기 디바이스, 즉 Software-Defined Accelerator(이하 SDA)는 아래 8가지 명령어로 사용할 수 있습니다.**

1. `moreh-smi -i` - SDA 활용 상태 모니터링하기
2. `moreh-smi -p` - SDA 상세 하드웨어 상태 모니터링하기
3. `moreh-smi -t` - SDA 토큰 정보 확인하기
4. `moreh-switch-model` - SDA 변경하기
5. `moreh-smi --reset` - SDA 프로세스 종료하기
6. `moreh-smi device --add` - SDA 생성하기
7. `moreh-smi device --rm` - SDA 삭제하기
8. `moreh-smi device --switch` - SDA 디바이스 기본값 변경하기

1~5번 명령어를 통해 단일 SDA 디바이스를 설정하고 모니터링 할 수 있으며, 6\~8번 명령어를 통해 다중 SDA 디바이스를 설정할 수 있습니다.

각 명령어의 다양한 옵션에 대해서 더 자세히 알고 싶다면 `moreh-smi --help` 로 확인 가능합니다.

**단일 SDA와 다중 SDA의 차이점**

MoAI Platform의 다중 SDA는 Token 1개당 1개 이상의 디바이스 종류를 생성/삭제할 수 있는 기능을 지원합니다. 하나 이상의 디바이스가 지원되는 것과 동시에 사용자 친화적으로 인터페이스가 구성되어 하나의 Token으로 여러 개의 디바이스의 프로세스를 유연하게 실행할 수 있습니다.

> 단일 SDA를 사용한다면 moreh-switch-model 명령어를 통해 하나의 GPU 자원의 종류를 선택할 수 있습니다. 반면에 다중 SDA를 사용한다면, 하나의 VM에서 여러 개의 SDA 디바이스를 동시에 선택하고 실행할 수 있습니다.
> 

하나의 VM에서 여러 개의 SDA 디바이스를 동시에 실행함으로써 아래와 같은 다양한 장점을 얻을 수 있습니다.

- VM 1개를 여러 명이 동시에 공유해야 할 경우, VM의 자원을 효율적으로 활용할 수 있습니다.
- 학습에 사용할 하이퍼파라미터를 탐색하기 위한 Hyperparameter Tuning 작업을 여러 번의 학습을 동시에 실행하여 최적의 설정 값을 찾을 수 있습니다.

위와 같은 상황에서 병렬 실행을 통해 동시에 GPU 자원을 사용할 수 있습니다.

예를 들어 아래와 같이 `moreh-smi device --add {model_id}`로 SDA를 추가하여 총 2개의 SDA가 설정되었다면 1개의 Token에 대해 VM 한 곳에서 동시에 2개의 프로세스를 실행할 수 있습니다.

```bash
(moreh) ubuntu@vm:~$ moreh-smi
14:23:17 September 12, 2023
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 23.9.0  Latest Version: 23.8.1  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  KT AI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
|    1     |  KT AI Accelerator  |  Large.256GB     |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+

```

**실행 프로세스**

`$ MOREH_VISIBLE_DEVICE=0 python train_your_script_00.py$ MOREH_VISIBLE_DEVICE=1 python train_your_script_01.py`

위 명령어로 여러 개의 GPU 묶음을 할당하여 병렬 학습을 진행할 수 있습니다.

- Device ID 0번 SDA → `train_your_script_00.py` 을 실행함과 동시에
- Device ID 1번 SDA → `train_your_script_01.py` 을 실행할 수 있습니다.

SDA 복제 기능(`Duplicable`) 설정을 통해 최대 횟수만큼 병렬 학습을 진행할 수 있으나 해당 기능은 관리자를 통해 설정 요청 부탁드립니다.

이제 개별 명령어에 대해 설명 드리겠습니다.

## 1. SDA 활용 상태 모니터링하기 `moreh-smi`

Moreh 소프트웨어 툴은 **`moreh-smi`** 명령어를 통해 현재 선택된 SDA 모델, 실행 중인 학습 프로세스, GPU Resource를 얼마나 할당받고 있는지를 확인할 수 있습니다.

```bash
(moreh) ubuntu@vm:~$ moreh-smi
13:42:32 September 12, 2023
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 23.9.0  Latest Version: 23.8.1  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  KT AI Accelerator  |  4xLarge.2048GB  |  886 MiB       |  2096640 MiB   |   62 %        |
|    1     |  KT AI Accelerator  |  -               |  -             |  -             |  -            |
|    2     |  KT AI Accelerator  |  -               |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+

Processes:
+-------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID   |        Process        |  Memory Usage  |
+=========================================================================+
|       0  |  901372  |  170394  |    python train.py    |  886 MiB       |
+-------------------------------------------------------------------------+

```

## 2. SDA token 정보 확인하기 `moreh-smi -p`

`moreh-smi -p` 명령어로 현재 선택된 SDA 모델에 할당된 노드의 아래와 같은 정보를 확인할 수 있습니다.

```yaml
- Dev Temp: GPU 온도
- Mem Temp: GPU 메모리 온도
- Dev Util: GPU 사용량
- Mem Util: GPU 메모리 사용량
- Mem Usage: GPU 메모리 활용도
```

```bash
(moreh) ubuntu@vm:~$ moreh-smi -p
13:42:36 September 12, 2023
+--------------------------------------------------------------------------------------+
|  Dev  |  Location  |  Dev Temp  |  Mem Temp  |  Dev Util  |  Mem Util  |  Mem Usage  |
+======================================================================================+
|    0  |  back01:0  |    36 C    |    43 C    |    35 %    |     0 %    |     1 %     |
|    1  |  back01:1  |    35 C    |    44 C    |    63 %    |     0 %    |     1 %     |
|    2  |  back01:2  |    39 C    |    50 C    |    66 %    |     0 %    |     1 %     |
|    3  |  back01:3  |    38 C    |    46 C    |    67 %    |     0 %    |     1 %     |
|    4  |  back01:4  |    38 C    |    49 C    |    82 %    |     0 %    |     1 %     |
|    5  |  back01:5  |    35 C    |    46 C    |    82 %    |     0 %    |     1 %     |
|    6  |  back01:6  |    42 C    |    49 C    |    75 %    |     0 %    |     1 %     |
|    7  |  back01:7  |    40 C    |    48 C    |    74 %    |     0 %    |     1 %     |
|    8  |  back02:0  |    37 C    |    47 C    |    99 %    |     0 %    |     1 %     |
|    9  |  back02:1  |    40 C    |    47 C    |    99 %    |     0 %    |     1 %     |
|   10  |  back02:2  |    42 C    |    50 C    |    99 %    |     0 %    |     1 %     |
|   11  |  back02:3  |    41 C    |    48 C    |    99 %    |     0 %    |     1 %     |
|   12  |  back02:4  |    41 C    |    48 C    |    99 %    |     0 %    |     1 %     |
|   13  |  back02:5  |    36 C    |    47 C    |    99 %    |     0 %    |     1 %     |
|   14  |  back02:6  |    41 C    |    49 C    |    99 %    |     0 %    |     1 %     |
|   15  |  back02:7  |    44 C    |    50 C    |    99 %    |     0 %    |     1 %     |
|   16  |  back32:0  |    38 C    |    46 C    |    85 %    |     0 %    |     1 %     |
|   17  |  back32:1  |    37 C    |    46 C    |    47 %    |     0 %    |     1 %     |
|   18  |  back32:2  |    42 C    |    51 C    |    48 %    |     0 %    |     1 %     |
|   19  |  back32:3  |    43 C    |    48 C    |    49 %    |     0 %    |     1 %     |
|   20  |  back32:4  |    38 C    |    48 C    |    72 %    |     0 %    |     1 %     |
|   21  |  back32:5  |    38 C    |    47 C    |    74 %    |     0 %    |     1 %     |
|   22  |  back32:6  |    44 C    |    48 C    |    70 %    |     0 %    |     1 %     |
|   23  |  back32:7  |    44 C    |    49 C    |    60 %    |     0 %    |     1 %     |
|   24  |  back75:0  |    40 C    |    44 C    |    77 %    |     0 %    |     1 %     |
|   25  |  back75:1  |    33 C    |    44 C    |    77 %    |     0 %    |     1 %     |
|   26  |  back75:2  |    37 C    |    46 C    |    62 %    |     0 %    |     1 %     |
|   27  |  back75:3  |    39 C    |    46 C    |    62 %    |     0 %    |     1 %     |
|   28  |  back75:4  |    32 C    |    43 C    |    53 %    |     0 %    |     1 %     |
|   29  |  back75:5  |    33 C    |    42 C    |    56 %    |     0 %    |     1 %     |
|   30  |  back75:6  |    40 C    |    46 C    |    72 %    |     0 %    |     1 %     |
|   31  |  back75:7  |    39 C    |    49 C    |    72 %    |     0 %    |     1 %     |
+--------------------------------------------------------------------------------------+

```

## 3. SDA token 정보 확인하기 `moreh-smi --token`

Token 값은 사용자를 식별하기 위한 해시 값이며 사용자마다 고유 값을 가지고 있습니다. Token은 일반적으로 **사용자의 가상 머신(VM) 안에 위치하며, 모레솔루션을 사용할 서버는 Token 값을 바탕으로 사용자를 식별하고 학습이 실행되므로,** Token 없이는 GPU 연산 및 Python 애플리케이션을 실행할 수 없습니다.

`moreh-smi --token` 또는 `moreh-smi -t` 명령어로 VM에서 Token 설정 상태를 확인할 수 있습니다.

터미널에 해당 명령어를 입력하면 어떤 Token이 설정되어있는지 확인할 수 있습니다.

```bash
(moreh) ubuntu@vm:~$ moreh-smi --token
15:23:58 June 14, 2023
+--------------------------------------------------------------------------+
|  Device  |        Name         |            Token           |    Model   |
+==========================================================================+
|  * 0     |      Moreh SDA      |  a3RfYTE2ODU1OTY3NzE2MTA=  |  12xlarge  |
+--------------------------------------------------------------------------+

```

## 4. SDA 변경하기 `moreh-switch-model`

`moreh-switch-model` 명령어를 통해 SDA 디바이스를 변경하여 가상 머신에서 사용할 GPU 리소스의 양을 조정할 수 있습니다.

`moreh-switch-model` 명령어를 사용하면 아래와 같은 입력창이 나타납니다.

```bash
(moreh) ubuntu@vm:~$ moreh-switch-model
Current KT AI Accelerator: Medium.128GB

1. Small.64GB
2. Medium.128GB  *
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

1~13 중 사용할 모델에 해당하는 정수를 입력하면 “The KT AI Accelerator model is successfully switched to {model_id}.” 메시지와 함께 입력된 디바이스 번호에 해당하는 SDA 모델로 변경됩니다. 여기서는 1번 `Small.64GB` 로 SDA 모델을 변경해보겠습니다.

```bash
Selection (1-13, q, Q): 1
The KT AI Accelerator model is successfully switched to  "Small.64GB".

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

변경을 계속하거나 `q` 또는 `Q`를 통해 SDA 모델 변경을 종료할 수 있습니다.

```bash
Selection (1-13, q, Q): q

(moreh) ubuntu@vm:~$
```

## 5. SDA 프로세스 종료하기 `moreh-smi --reset`

`moreh-smi --reset` 또는 `moreh-smi -r`  명령어를 통해 SDA 디바이스를 사용하고 있는 프로세스를 종료할 수 있습니다.

다음은 학습 중 종료한 예시입니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-smi
16:11:38 September 11, 2023
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 23.9.0  Latest Version: 23.8.1  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  KT AI Accelerator  |  4xLarge.2048GB  |  655 MiB       |  2096640 MiB   |    0 %        |
|    1     |  KT AI Accelerator  |  -               |  -             |  -             |  -            |
|    2     |  KT AI Accelerator  |  -               |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+

Processes:
+-----------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID   |          Process          |  Memory Usage  |
+=============================================================================+
|       0  |  900955  |  167952  |  python pytorch_mnist.py  |  655 MiB       |
+-----------------------------------------------------------------------------+

(pytorch) ubuntu@vm:~$ moreh-smi --reset
Device release success.

```

“Device release success.” 메시지와 함께 종료된 걸 확인할 수 있습니다.

아래와 같이 프로세스가 존재하지 않는 경우에는 “Device release failed. (Not running job.)” 메시지와 함께 실패합니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-smi
16:11:37 September 11, 2023
+-----------------------------------------------------------------------------------------------------+
|                                                **Current** Version: 23.9.0  Latest Version: 23.8.1  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+====================================****=============================================================+
|  * 0     |  KT AI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
|    1     |  KT AI Accelerator  |  -               |  -             |  -             |  -            |
|    2     |  KT AI Accelerator  |  -               |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
(pytorch) ubuntu@vm:~$ moreh-smi --reset
Device release failed. (Not running job.)

```

## 6. SDA 추가하기 `moreh-smi device --add`

- 가상 머신(VM)이 생성된 직후에는 SDA는 1개까지만 기본값으로 제한되어 있습니다. **2개 이상의 SDA 사용이 필요한 경우 관리자에게 문의하여 제한값 설정을 요청 부탁드립니다.**
- 하나의 VM 내에서는 최대 5개까지의 SDA를 생성할 수 있습니다.

Token의 제한 값이 변경된 이후 `moreh-smi device --add` 명령어로 SDA를 추가할 수 있습니다.

다음은 SDA를 추가하는 예제입니다.

```bash
(moreh) ubuntu@vm:~$ moreh-smi device --add
1. Small.64GB
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

`moreh-smi device --add` 커맨드를 입력하면 `moreh-switch-model`과 동일한 인터페이스가 나타납니다. 1~13 중 사용할 모델에 해당하는 정수를 입력하면 “Create device success.” 메시지와 함께 입력된 디바이스 번호에 해당하는 SDA가 생성됩니다. 여기서는 3번 `Large.256GB` 로 SDA를 하나 더 생성해 보겠습니다.

```bash
Selection (1-13, q, Q): 3
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |  KT AI Accelerator  |  4xLarge.2048GB  |
|    1     |  KT AI Accelerator  |  Large.256GB     |
+---------------------------------------------------+
Create device success.
1. Small.64GB
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

```

`moreh-smi device --add {model_id}` 명령어를 통해 대화형 입력창 없이 바로 SDA를 생성할 수도 있습니다.

여기서 `{model_id}` 는 SDA 모델의 번호를 의미하며, `Large.256GB` 의 경우에는 ‘3’ 이 됩니다.

## 7. 생성된 SDA 디바이스 삭제하기 `moreh-smi device --rm`

생성된 SDA 디바이스를 삭제하려면 `moreh-smi device --rm` 명령어를 사용하면 됩니다.

`moreh-smi device --rm {Device_ID}` 명령어로 특정 Device_ID에 해당하는 SDA를 삭제해보겠습니다.

```bash
(moreh) ubuntu@vm:~/sample$ moreh-smi
14:23:17 September 12, 2023
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 23.9.0  Latest Version: 23.8.1  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  KT AI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
|    1     |  KT AI Accelerator  |  Large.256GB     |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
(moreh) ubuntu@vm:~$ moreh-smi --rm 1
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |  KT AI Accelerator  |  4xLarge.2048GB  |
+---------------------------------------------------+
Remove device success.

```

만약 help message에 `device --add` 와 같은 옵션의 도움말이 등장하지 않는다면 사용자 token에 대한 최대 디바이스 개수가 1로 설정된 것이므로 고객지원을 요청 부탁드립니다.

SSH 클라이언트와 통신이 끊겨 학습이 종료되는 문제를 방지하기 위하여, tmux 등의 터미널 다중화 프로그램을 이용하시는 것을 권장드립니다.

## 8. SDA 생성된 디바이스 기본값 변경하기

**`moreh-smi device --switch {Device_ID}`** 명령어를 입력하면 이미 생성된 디바이스의 ID에 해당하는 디바이스로 변경됩니다.

`moreh-smi device --switch {Device_ID}`를 통해 0번 `Medium.128GB`을 기본 SDA로 변경해 보겠습니다.

```
(pytorch) ubuntu@vm:~$ moreh-smi device --switch 0
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |  KT AI Accelerator  |  4xLarge.2048GB  |
|    1     |  KT AI Accelerator  |  Medium.128GB    |
|    2     |  KT AI Accelerator  |  Medium.128GB    |
+---------------------------------------------------+
Switch current device success.

```

생성된 SDA 모델 리스트 중 디바이스에 해당하는 정수를 입력하면 “Switch current device success.” 메시지와 함께 입력된 SDA가 기본 디바이스로 설정됩니다. 학습 프로세스 실행하면 설정한 기본 SDA 디바이스를 사용합니다.

여기서는 다시 1번 `Medium.128GB`을 기본 SDA 디바이스로 변경해 보겠습니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-smi device --switch 1
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|    0     |  KT AI Accelerator  |  4xLarge.2048GB  |
|  * 1     |  KT AI Accelerator  |  Medium.128GB    |
|    2     |  KT AI Accelerator  |  Medium.128GB    |
+---------------------------------------------------+
Switch current device success.

```

이제 학습 실행 시 기본 값으로 1번 디바이스를 사용하게 됩니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-smi
13:53:04 September 12, 2023
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 23.9.0  Latest Version: 23.8.1  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |  KT AI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
|  * 1     |  KT AI Accelerator  |  Medium.128GB    |  56 MiB        |  131040 MiB    |    0 %        |
|    2     |  KT AI Accelerator  |  -               |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+

Processes:
+-----------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID   |          Process          |  Memory Usage  |
+=============================================================================+
|       1  |  901378  |  170610  |  python pytorch_mnist.py  |  56 MiB        |
+-----------------------------------------------------------------------------+

```