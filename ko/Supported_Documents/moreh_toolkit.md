---
icon: tools
tags: [guide]
order: 110
---

# MoAI Platform의 toolkit 사용하기

## Moreh Toolkit의 기능

Moreh Toolkit은 MoAI Platform 상에서 MoAI Accelerator를 관리하거나 모니터링할 때 유용한 command line 도구입니다. 사용자는 Moreh Toolkit 는 현재 3가지의 command line 명령어를 제공하고 있으며 (`moreh-smi`, `moreh-switch-model`, `update-moreh`), 이를 사용하여 MoAI Accelerator 를 관리 및 변경할 수 있고, 또한 현재 유저의 환경에 설치된 모레 솔루션을 손쉽게 업데이트할 수 있습니다.

Moreh Toolkit의 주요 기능은 다음과 같습니다: 

- MoAI Accelerator의 메모리 사용량 및 프로세스 현황을 실시간으로 확인하고 모니터링
- AI 가속기를 변경하고 프로세스를 실행하여 최적의 성능을 얻고자 할 때
- Moreh 솔루션을 최신 버전으로 업데이트하거나 롤백해야 할 때

## `moreh-smi`

`moreh-smi`는 사용자가 MoAI Accelerator를 관리하고 모니터링할 수 있는 명령어 입니다. MoAI Platform Pytorch가 설치된 conda 환경에서 다음과 같이 실행할 수 있습니다.

```jsx
$ moreh-smi
15:51:25 April 29, 2024 
+----------------------------------------------------------------------------------------------+
|                                                   Current Version:   Latest Version: 24.5.0  |
+----------------------------------------------------------------------------------------------+
|  Device  |        Name         |   Model   |  Memory Usage  |  Total Memory  |  Utilization  |
+==============================================================================================+
|  * 0     |  MoAI Accelerator   |  4xlarge  |  -             |  -             |  -            |
+----------------------------------------------------------------------------------------------+
```

현재 MoAI Accelerator 를 사용하여 학습을 진행하고 있다면, 다른 터미널 세션을 활용하여 `moreh-smi` 을 실행할 경우 실행중인 프로세스 정보를 다음과 같은 화면을 보실 수 있습니다. 또한 `moreh-smi` 을 활용하면 현재 본인의 Job ID 를 확인할 수 있으므로, MoAI Platform 에서 학습 또는 추론에 문제가 생길 경우 해당 Job ID 와 함께 고객지원을 문의하면 더 빠르게 응답을 받으실 수 있습니다.

```jsx
$ moreh-smi
17:58:15 April 29, 2024 
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  xLarge.512GB    |  397 MiB       |  524160 MiB    |    0 %        |
+-----------------------------------------------------------------------------------------------------+

Processes:
+-----------------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID    |             Process            |  Memory Usage  |
+===================================================================================+
|       0  |  976356  |  1548305  |  python tutorial/train_gpt.py  |  397 MiB       |
+-----------------------------------------------------------------------------------+
```

## MoAI Accelerator 의 Multi Accelerator 기능 활용하기

유저가 별도의 세팅을 하지 않을 경우에는 기본적으로 하나의 SSH 환경에 하나의 MoAI Accelerator 만 존재할 것입니다. 기본적으로 MoAI Accelerator 한 개로는 하나의 프로세스만 실행할 수 있기 때문에, 기본 세팅으로는 하나의 SSH 환경에서 하나의 프로세스 실행만 가능합니다.

하지만 경우에 따라서는 하나의 SSH 환경에서도 여러 개 MoAI Accelerator 를 활용하여 동시에 여러 개의 프로세스로 학습을 실행하고 싶은 경우도 있을 겁니다. (예: 동일한 소스코드이나 하이퍼 파라미터를 변경해서 여러 개의 학습 실험을 동시에 수행하고 싶은 경우) 이런 경우 `moreh-smi` 에서 하나의 토큰 내에 여러 개의 MoAI Accelerator 을 생성하면, 동시에 여러개의 프로세스를 수행할 수 있습니다.

다음 예제를 통해 AI 가속기를 추가, 변경 삭제해보겠습니다.

### 1. AI 가속기 추가하기

먼저 AI 가속기를 추가해보겠습니다. 2개 이상의 AI 가속기를 사용하기 위해서`moreh-smi device --add` 커멘드를 입력하면 아래와 같은 인터페이스가 나타납니다.

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

1~13 중 사용할 모델에 해당하는 정수를 입력하면 “Create device success.” 메시지와 함께 입력된 디바이스 번호에 해당하는 AI 가속기가 생성됩니다. 하나의 VM 내에서는 최대 5개 AI가속기를 생성할 수 있습니다.

아래 예제에서는 10번 `8xLarge.4096GB` AI 가속기를 추가해 보겠습니다.

```bash
Selection (1-13, q, Q): 10
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |  AI Accelerator  |  Large.256GB        |
|    1     |  AI Accelerator  |  8xLarge.4096GB     |
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

### 2. AI 가속기 기본값 변경하기

`moreh-smi device --switch {Device_ID}` 는 기본값으로 설정된 MoAI Accelerator 를 변경할 수 있는 명령어 입니다.

다음과 같이 사용할 수 있습니다 : 

```bash
(moreh) ubuntu@vm:~$ moreh-smi device --switch 1

+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|    0     |  KT AI Accelerator  |  2xLarge.1024GB  |
|  * 1     |  KT AI Accelerator  |  xLarge.512GB    |
|    2     |  KT AI Accelerator  |  2xLarge.1024GB  |
|    3     |  KT AI Accelerator  |  8xLarge.4096GB  |
|    4     |  KT AI Accelerator  |  Small.64GB      |
+---------------------------------------------------+
Switch Current Device success.
```

 현재 기본값으로 설정된 MoAI Accelerator가 1번 가속기로 변경된 것을 확인할 수 있습니다.

```bash
Selection (0-4, q, Q): q

(moreh) ubuntu@vm:~$ moreh-smi
10:49:12 May 07, 2024
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.2.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |  KT AI Accelerator  |  2xLarge.1024GB  |  -             |  -             |  -            |
|  * 1     |  KT AI Accelerator  |  xLarge.512GB    |  -             |  -             |  -            |
|    2     |  KT AI Accelerator  |  2xLarge.1024GB  |  -             |  -             |  -            |
|    3     |  KT AI Accelerator  |  8xLarge.4096GB  |  -             |  -             |  -            |
|    4     |  KT AI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

### 3. AI 가속기 변경하기

`moreh-switch-model` 는 현재 설정된 MoAI Accelerator 의 flavor(가속기 사양)를 변경할 수 있는 툴입니다. MoAI Accelerator 의 flavor를 변경함으로써 GPU 메모리를 얼만큼 사용할 것인지 결정합니다. 

다음과 같이 사용할 수 있습니다 : 

예를 들어, `moreh-smi` 명령어의 결과가 다음과 같다면 이는 “현재 기본값으로 설정된 MoAI Accelerator는 0번 가속기이며 이 MoAI Accelerator의 유형은 `Small.64GB` 모델”이라는 의미입니다. 

```jsx
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
|    1     | MoAI Accelerator    |  Medium.128GB    |  -             |  -             |  -            |
|    2     | MoAI Accelerator    |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

`moreh-switch-model` 명령어를 사용하면 아래와 같은 입력창이 나타납니다.

```bash
(moreh) ubuntu@vm:~$ moreh-switch-model
Current AI Accelerator: Medium.128GB

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

1~13 중 사용할 모델에 해당하는 정수(디바이스 번호)를 입력하면 “The MoAI Platform AI Accelerator model is successfully switched to {model_id}.” 메시지와 함께 입력된 디바이스 번호에 해당하는 MoAI Accelerator로 변경됩니다. 

지금은 3번 `Large.256GB` 로 MoAI Accelerator를 변경해보겠습니다.

```bash
Selection (1-13, q, Q): 3
The AI Accelerator model is successfully switched to  "Large.256GB".

1. Small.64GB  
2. Medium.128GB
3. Large.256GB *
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

변경을 계속하거나 `q` 또는 `Q`를 통해 MoAI Accelerator 변경을 종료할 수 있습니다.

변경이 완료된 후 다시 `moreh-smi` 를 사용하여 확인한 결과는 다음과 같습니다. 

```jsx
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.2.0  Latest Version: 24.2.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     | MoAI Accelerator    |  Large.256GB     |  -             |  -             |  -            |
|    1     | MoAI Accelerator    |  Medium.128GB    |  -             |  -             |  -            |
|    2     | MoAI Accelerator    |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

0번 `Small.64GB` 모델 유형의 MoAI Accelerator가 `Large.256GB` 모델 유형으로 변경된 것을 확인할 수 있습니다. 

### 3. AI 가속기 삭제하기

이번에는 생성된 디바이스를 `moreh-smi device --rm {Device_ID}`커멘드로 특정 디바이스 ID에 해당하는 가속기를 삭제해보겠습니다.

```bash
(moreh) ubuntu@vm:~$ moreh-smi --rm 1
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |  AI Accelerator  |    Large.256GB     |
+---------------------------------------------------+
Remove device success.
```

위와 같은 커멘드를 입력해서 Device ID가 1인 AI 가속기인 `8xLarge.4096GB` 가 삭제되었습니다. 확인을 위해 다시 moreh-smi를 실행하면 해당 디바이스가 삭제된 것을 확인할 수 있습니다.

    

---

```bash
$ moreh-smi
18:02:58 April 29, 2024 
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |  MoAI Accelerator   |  8xLarge.4096GB  |  2340037 MiB   |  4193280 MiB   |   87 %        |
|  * 1     |  MoAI Accelerator   |  4xLarge.2048GB  |  789281 MiB    |  2096640 MiB   |   53 %        |
+-----------------------------------------------------------------------------------------------------+

Processes:
+---------------------------------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID    |                     Process                    |  Memory Usage  |
+===================================================================================================+
|       0  |  976362  |  1552601  |  python tutorial/train_gpt.py --batch-size 16  |  2340037 MiB   |
|       1  |  976366  |  1553195  |  python tutorial/train_gpt.py --batch-size 8   |  789281 MiB    |
+---------------------------------------------------------------------------------------------------+
```

## 그 외의 다양한 옵션 활용하기

`moreh-smi` 는 이외에도 다양한 다양한 옵션을 제공합니다. 다음과 같이 `--help` 옵션을 활용하면 어떠한 옵션이 제공되는지 확인할 수 있습니다.

```jsx
$ moreh-smi --help

Usage: moreh-smi [-h | --help] [-r | --reset] [-s | --server-version] [-v | --version] [-t | --token] [-i | --idx]
                 [device {--add [model_id] | --rm [device_id] | --switch [device_id]}]

Basic Options:
  -h, --help             provide information about available command switches and their options
  -r, --reset            stop the running process
  -s, --server-version   print Moreh Framework version information
  -v, --version          print current software version information
  -t, --token            print Moreh Solution token information
  -i, --idx              select a device to print

Device Options:
  device --list                 list available models for adding device
  device --add [model_id]       add a device corresponding to model_id
  device --rm  [device_id]      remove a device corresponding to device_id
  device --switch [device_id]   switch to the device corresponding to [device_id]

Device Options operate interactively if there are no optional arguments([model_id], [device_id]).

Device Example:
  moreh-smi device --list
  moreh-smi device --add
  moreh-smi device --add 2
  moreh-smi device --switch 1
  moreh-smi -i 2
```

(위 내용들이 잘 정리가 되었다면, 굳이 아래 중에 1, 5, 6, 7, 8 은 언급하지 않아도 될 것 같습니다)

1. `moreh-smi` -  MoAI Accelerator 활용 상태 모니터링하기
2. `moreh-smi -p` -  MoAI Accelerator 상세 하드웨어 상태 모니터링하기
3. `moreh-smi -t` -  MoAI Accelerator 토큰 정보 확인하기
4. `moreh-smi --reset` -  MoAI Accelerator 프로세스 종료하기
5. `moreh-smi device --list` - 추가 가능한 MoAI Accelerator 종류 확인하기
6. `moreh-smi device --add` -  새로운 MoAI Accelerator추가하기
7. `moreh-smi device --rm` -  MoAI Accelerator 삭제하기
8. `moreh-smi device --switch` - MoAI Accelerator 디바이스 기본값 변경하기

# update-moreh

`update-moreh`는 conda 환경을 새롭게 생성하고 그 위에 모레 솔루션을 설치하거나, 이미 conda 환경에 설치된 모레 솔루션의 버전을 업데이트할 수 있는 명령어입니다. 다음과 같은 상황에서 `update-moreh` 를 사용할 수 있습니다. 

- 새롭게 conda 환경을 생성한 경우 아직 모레 솔루션에 필요한 Python 패키지 설치가 필요합니다. 이 경우에는 `update-moreh` 명령어를 통해서 최신 버전의 모레 솔루션을 간단하게 설치할 수 있습니다.

```jsx
$ conda create --name my_env python=3.8
$ update-moreh
```

- 이미 모레 솔루션이 설치된 conda 환경 내에서도 최신 버전의 모레 솔루션을 사용하고자 할 때, `update-moreh` 명령어를 단독으로 사용하여 현재 사용 중인 모레 솔루션을 최신버전으로 업데이트 할 수 있습니다.

```jsx
$ update-moreh # Latest 버전으로 업데이트
```

- 필요에 따라 특정 버전의 모레 솔루션을 설치해야 할 경우가 있습니다. 이 경우에는 `--target` 옵션을 사용하여 사용자가 설치하고 싶은 특정 버전을 지정할 수 있습니다.

```jsx
update-moreh --target 24.5.301 # 24.5.301 버전으로 설치
update-moreh --target 24.5.302 # 24.5.302 버전으로 설치
```

- conda 환경에서 다른 패키지간의 의존성 충돌이 발생하는 문제 등으로 인해 모레 솔루션이 정상적으로 동작하지 않는 경우, conda 환경을 재구성을 해야 할 수 있습니다. 이러한 경우에도 conda 환경 내의 모레 솔루션 복구를 위하여 `update-moreh` 를 사용할 수 있습니다. 후자의 경우 `--force` 옵션을 사용하여 환경 재구성이 가능합니다. (`—-target` 옵션과 같이 사용 가능)

```bash
update-moreh --force --target 24.5.301
```

# moreh-switch-model

`moreh-switch-model` 는 현재 기본값으로 설정된 MoAI Accelerator 의 flavor를 변경할 수 있는 툴입니다. MoAI Accelerator 의 flavor를 변경함으로써 GPU 메모리를 얼만큼 사용할 것인지 결정합니다. 

다음과 같이 사용할 수 있습니다 : 

예를 들어, `moreh-smi` 명령어의 결과가 다음과 같다면 이는 “현재 기본값으로 설정된 MoAI Accelerator는 0번 가속기이며 이 MoAI Accelerator의 유형은 `Small.64GB` 모델”이라는 의미입니다. 

```jsx
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
|    1     | MoAI Accelerator    |  Medium.128GB    |  -             |  -             |  -            |
|    2     | MoAI Accelerator    |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

이를 다른 유형의 모델로 변경하고자 할 때, `moreh-switch-model` 을 사용할 수 있습니다. 

`moreh-switch-model` 명령어를 사용하면 아래와 같은 입력창이 나타납니다.

```bash
(moreh) ubuntu@vm:~$ moreh-switch-model
Current AI Accelerator: Medium.128GB

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

1~13 중 사용할 모델에 해당하는 정수(디바이스 번호)를 입력하면 “The MoAI Platform AI Accelerator model is successfully switched to {model_id}.” 메시지와 함께 입력된 디바이스 번호에 해당하는 MoAI Accelerator로 변경됩니다. 

지금은 3번 `Large.256GB` 로 MoAI Accelerator를 변경해보겠습니다.

```bash
Selection (1-13, q, Q): 3
The AI Accelerator model is successfully switched to  "Large.256GB".

1. Small.64GB  
2. Medium.128GB
3. Large.256GB *
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

변경을 계속하거나 `q` 또는 `Q`를 통해 MoAI Accelerator 변경을 종료할 수 있습니다.

변경이 완료된 후 다시 `moreh-smi` 를 사용하여 확인한 결과는 다음과 같습니다. 

```jsx
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.2.0  Latest Version: 24.2.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     | MoAI Accelerator    |  Large.256GB     |  -             |  -             |  -            |
|    1     | MoAI Accelerator    |  Medium.128GB    |  -             |  -             |  -            |
|    2     | MoAI Accelerator    |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

0번 `Small.64GB` 모델 유형의 MoAI Accelerator가 `Large.256GB` 모델 유형으로 변경된 것을 확인할 수 있습니다. 

---