---
icon: tools
tags: [guide, moreh_toolkit]
order: 40
---

# Moreh Toolkit 가이드

Moreh Toolkit은 MoAI Platform 상에서 [MoAI Accelerator](https://docs.moreh.io/ko/moai_features/virtualization/#gpu-%EA%B0%80%EC%83%81%ED%99%94-moai-accelerator)를 관리하거나 모니터링할 때 유용한 command line 도구입니다.  이 도구는 사용자에게 세 가지 명령어 (`moreh-smi`, `moreh-switch-model`, `update-moreh`)를 제공하여 MoAI Accelerator를 효율적으로 관리하고, 설치된 MoAI Platform을 손쉽게 업데이트할 수 있도록 합니다.

## 주요 기능

Moreh Toolkit의 주요 기능은 다음과 같습니다: 

1. **MoAI Accelerator의 모니터링:**
    - **`moreh-smi`** 명령어를 사용하여 MoAI Accelerator에 할당된 메모리 사용량 및 프로세스 현황을 실시간으로 확인할 수 있습니다.
2. **MoAI Accelerator 변경:**
    - **`moreh-switch-model`** 명령어를 사용하여 MoAI Accelerator를 변경하고 최적의 성능을 얻기 위한 프로세스를 실행할 수 있습니다.
3.  **MoAI Platform 버전 설치 및 업데이트:**
    - **`update-moreh`** 명령어를 통해 MoAI Platform을 최신 버전으로 설치 및 업데이트하거나 필요시 이전 버전으로 롤백할 수 있습니다.


## MoAI Accelerator 모니터링: `moreh-smi` 

`moreh-smi`는 사용자가 MoAI Accelerator를 관리하고 모니터링할 수 있는 명령어 입니다. MoAI Platform Pytorch가 설치된 conda 환경에서 다음과 같이 실행할 수 있습니다.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  4xlarge.2048GB  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

현재 MoAI Accelerator를 사용하여 학습을 진행하고 있다면, 다른 터미널 세션을 활용하여 `moreh-smi` 을 실행할 경우 실행중인 프로세스 정보를 다음과 같은 화면을 보실 수 있습니다. 또한 `moreh-smi` 을 활용하면 현재 본인의 Job ID 를 확인할 수 있으므로, MoAI Platform 에서 학습 또는 추론에 문제가 생길 경우 해당 Job ID 와 함께 고객지원을 문의하면 더 빠르게 응답을 받으실 수 있습니다. 

아래 예시에서 확인할 수 있는 Job ID는 976356입니다.

```bash
$ moreh-smi
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

### MoAI Accelerator 의 Multi Accelerator 기능 활용하기

유저가 별도의 세팅을 하지 않을 경우에는 기본적으로 하나의 VM 또는 컨테이너 환경에 하나의 MoAI Accelerator만 존재할 것입니다. 기본적으로 MoAI Accelerator 한 개로는 하나의 프로세스만 실행할 수 있기 때문에, 기본 세팅으로는 하나의 환경에서 하나의 프로세스 실행만 가능합니다.

하지만 경우에 따라서는 하나의 환경에서도 여러 개 MoAI Accelerator를 활용하여 동시에 여러 개의 프로세스로 학습을 실행하고 싶은 경우도 있을 겁니다. (예: 동일한 소스코드이나 하이퍼 파라미터를 변경해서 여러 개의 학습 실험을 동시에 수행하고 싶은 경우) 이런 경우 `moreh-smi` 에서 하나의 토큰 내에 여러 개의 MoAI Accelerator를 생성하면, 동시에 여러개의 프로세스를 수행할 수 있습니다.
다음 예제를 통해 MoAI Accelerator를 추가, 변경 삭제해보겠습니다.

### MoAI Accelerator 추가하기 `moreh-smi device --add`

먼저 MoAI Accelerator를 추가해보겠습니다. 2개 이상의 MoAI Accelerator를 사용하기 위해서`moreh-smi device --add` 커멘드를 입력하면 아래와 같은 인터페이스가 나타납니다.

```bash
$ moreh-smi device --add
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

1~13 중 사용할 모델에 해당하는 정수를 입력하면 “Create device success.” 메시지와 함께 입력된 디바이스 번호에 해당하는 MoAI Accelerator가 생성됩니다. 하나의 환경 내에서는 최대 5개 AI가속기를 생성할 수 있습니다. 더 많은 MoAI Accelerator 를 생성하려면 인프라 관리자에게 문의하십시오.

아래 예제에서는 10번 `8xLarge.4096GB` MoAI Accelerator를 추가해 보겠습니다.


```bash
$ moreh-smi device --add 10
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |   MoAI Accelerator  |  xLarge.512GB    |  -             |  -             |  -            |
|  * 1     |   MoAI Accelerator  |  8xLarge.4096GB  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

---

### MoAI Accelerator 기본값 변경하기: `moreh-smi device --switch`

`moreh-smi device --switch {Device_ID}` 는 기본값으로 설정된 MoAI Accelerator를 변경할 수 있는 명령어 입니다.

다음과 같이 사용할 수 있습니다 :

```bash
$ moreh-smi device --switch 1
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|    0     |   MoAI Accelerator  |  2xLarge.1024GB  |
|  * 1     |   MoAI Accelerator  |  xLarge.512GB    |
|    2     |   MoAI Accelerator  |  2xLarge.1024GB  |
|    3     |   MoAI Accelerator  |  8xLarge.4096GB  |
|    4     |   MoAI Accelerator  |  Small.64GB      |
+---------------------------------------------------+
Switch Current Device success.
```

 현재 기본값으로 설정된 MoAI Accelerator가 1번 가속기로 변경된 것을 확인할 수 있습니다.

```bash
Selection (0-4, q, Q): q

$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |   MoAI Accelerator  |  2xLarge.1024GB  |  -             |  -             |  -            |
|  * 1     |   MoAI Accelerator  |  xLarge.512GB    |  -             |  -             |  -            |
|    2     |   MoAI Accelerator  |  2xLarge.1024GB  |  -             |  -             |  -            |
|    3     |   MoAI Accelerator  |  8xLarge.4096GB  |  -             |  -             |  -            |
|    4     |   MoAI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

### MoAI Accelerator 삭제하기: `moreh-smi device --rm`

이번에는 생성된 디바이스를 `moreh-smi device --rm {Device_ID}`커멘드로 특정 디바이스 ID에 해당하는 가속기를 삭제해보겠습니다.

다음과 같이 사용할 수 있습니다 : 


```bash
$ moreh-smi --rm 1
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |   MoAI Accelerator  |    Large.256GB   |
+---------------------------------------------------+
Remove device success.
```


위와 같은 커멘드를 입력해서 Device ID가 1인 MoAI Accelerator인 [!badge variant="secondary" text="8xLarge.4096GB"] 가 삭제되었습니다. 확인을 위해 다시 `moreh-smi`를 실행하면 해당 디바이스가 삭제된 것을 확인할 수 있습니다.

### 그 외의 다양한 옵션 활용하기 `--help`

`moreh-smi` 는 이외에도 다양한 다양한 옵션을 제공합니다. 다음과 같이 `--help` 옵션을 활용하면 어떠한 옵션이 제공되는지 확인할 수 있습니다.

```
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

- `moreh-smi -p` -  MoAI Accelerator 상세 하드웨어 상태 모니터링하기
- `moreh-smi -t` -  MoAI Accelerator토큰 정보 확인하기

!!!info 
학습 도중에 프로세스가 꼬이거나 종료가 잘 안되어서 Process Running 과 같은 메시지가 출력된다면 `moreh-smi --reset` 명령어를 활용하세요.
!!!

-----
## MoAI Accelerator 변경하기: `moreh-switch-model`

`moreh-switch-model` 는 현재 설정된 MoAI Accelerator의 flavor(가속기 사양)를 변경할 수 있는 툴입니다. MoAI Accelerator의 flavor를 변경함으로써 GPU 메모리를 얼만큼 사용할 것인지 결정합니다. 

다음과 같이 사용할 수 있습니다 : 

예를 들어, `moreh-smi` 명령어의 결과가 다음과 같다면 이는 “현재 기본값으로 설정된 MoAI Accelerator는 0번 가속기이며 이 MoAI Accelerator의 유형은 [!badge variant="secondary" text=Small.64GB] 모델”이라는 의미입니다.

```bash
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |   MoAI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
|    1     |   MoAI Accelerator  |  Medium.128GB    |  -             |  -             |  -            |
|    2     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     |   MoAI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

`moreh-switch-model` 명령어를 사용하면 아래와 같은 입력창이 나타납니다.

```bash
$ moreh-switch-model
Current MoAI Accelerator: Medium.128GB

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

1~13 중 사용할 모델에 해당하는 정수(디바이스 번호)를 입력하면 “`The MoAI Platform AI Accelerator model is successfully switched to {model_id}.”` 메시지와 함께 입력된 디바이스 번호에 해당하는 MoAI Accelerator로 변경됩니다. 

지금은 3번 [!badge variant="secondary" text=Large.256GB]로 MoAI Accelerator를 변경해보겠습니다.

```bash
Selection (1-13, q, Q): 3
The MoAI Accelerator model is successfully switched to  "Large.256GB".

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

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |   MoAI Accelerator  |  Large.256GB     |  -             |  -             |  -            |
|    1     |   MoAI Accelerator  |  Medium.128GB    |  -             |  -             |  -            |
|    2     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     |   MoAI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

0번 [!badge variant="secondary" text=Small.64GB] 모델 유형의 MoAI Accelerator가 [!badge variant="secondary" text=Large.256GB]모델 유형으로 변경된 것을 확인할 수 있습니다.

------

## MoAI Platform 업데이트 하기: `update-moreh`

`update-moreh`는 conda 환경을 새롭게 생성하고 그 위에 MoAI Platform을 설치하거나, 이미 conda 환경에 설치된 MoAI Platform의 버전을 업데이트할 수 있는 명령어입니다. 다음과 같은 상황에서 `update-moreh` 를 사용할 수 있습니다. 

- 새롭게 conda 환경을 생성한 경우 아직 MoAI Platform에 필요한 Python 패키지 설치가 필요합니다. 이 경우에는 `update-moreh` 명령어를 통해서 최신 버전의 MoAI Platform을 간단하게 설치할 수 있습니다.

```bash
$ conda create --name my_env python=3.8
$ conda activate my_env
$ update-moreh # MoAI Platform 설치

Do you want to proceed? (y/n, default:n)
y
Moreh Framework installation start...
Moreh Framework installation successful.
```
- 이미 MoAI Platform이 설치된 conda 환경 내에서도 최신 버전의 MoAI Platform을 사용하고자 할 때, `update-moreh` 명령어를 단독으로 사용하여 현재 사용 중인 MoAI Platform을 최신버전으로 업데이트 할 수 있습니다.

```bash
$ update-moreh # Latest 버전으로 업데이트
Currently installed: 
Possible upgrading version: 24.5.0

Do you want to upgrade? (y/n, default:n)
y
Moreh Framework installation start...
Moreh Framework installation successful.
$ update-moreh
Already installed : 24.5.0
```

- 필요에 따라 특정 버전의 MoAI Platform을 설치해야 할 경우가 있습니다. 이 경우에는 `--target` 옵션을 사용하여 사용자가 설치하고 싶은 특정 버전을 지정할 수 있습니다.

```bash
update-moreh --target 24.5.0 # 24.5.0 버전으로 설치
```

- conda 환경에서 다른 패키지간의 의존성 충돌이 발생하는 문제 등으로 인해 MoAI Platform이 정상적으로 동작하지 않는 경우, conda 환경을 재구성을 해야 할 수 있습니다. 이러한 경우에도 conda 환경 내의 MoAI Platform 복구를 위하여 **`update-moreh`** 를 사용할 수 있습니다. 후자의 경우 `--force` 옵션을 사용하여 환경 재구성이 가능합니다. (**`—-target`** 옵션과 같이 사용 가능)

```bash
update-moreh --force --target 24.5.0
```
