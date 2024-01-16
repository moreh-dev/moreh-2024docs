---
icon: cloud
tags: [guide]
order: 90
---

# Docker 이미지로 Moreh 실행하기 

### MoAI Platform에서 Docker로 Moreh 솔루션을 실행하는 방법

### `moreh-docker-run`

MoAI Platform은 도커 컨테이너 안에서 AI 가속기를 사용하는 PyTorch 프로그램을 실행할 수 있도록 전용 도커 이미지를 제공하고 있습니다. VM에서 다음의 명령어들을 실행하여 AI 가속기가 활성화된 컨테이너를 실행할 수 있습니다.


`moreh-docker-run` 은 도커의 권한이 필요한 실행 스크립트입니다. 따라서 아래와 같은 명령어로 사전에 docker 권한 수정이 필요합니다.
 `sudo chmod 666 /var/run/docker.sock`

```bash
$ ubuntu@vm:~$ moreh-docker-run
Checking for nvidia-container-toolkit: install ok installed
WARNING! Using --password via the CLI is insecure. Use --password-stdin.
WARNING! Your password will be stored unencrypted in /home/ubuntu/.docker/config.json.
Configure a credential helper to remove this warning. See
<https://docs.docker.com/engine/reference/commandline/login/#credentials-store>

Login Succeeded
Unable to find image 'sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1' locally
23.9.0-pytorch1.7.1: Pulling from moreh
99803d4b97f3: Pull complete
c38a5999a5b8: Pull complete
b053b2940d34: Pull complete
71851d75261e: Pull complete
850dae3da5a5: Pull complete
4b44d5d42702: Pull complete
Digest: sha256:0f5b358a780a656d658b6fc0bdca1064933612d7ec8bbb941833a1ba480bd05d
Status: Downloaded newer image for sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1

```

`moreh-docker-run` 명령어를 사용해 Moreh 솔루션이 담긴 도커 이미지를 실행합니다. 추가적으로 다른 옵션값을 안주고 실행했을 경우에는 현재까지 배포된 Moreh 솔루션 이미지 중 가장 최신 버전 도커 이미지를 실행하게 됩니다.

`moreh-docker-run` 명령어 뒤에 추가 옵션을 통해 도커 이미지만 다운로드 받기, 버전 확인 등을 실행할 수 있습니다.

`moreh-docker-run` 명령어는 **Moreh 솔루션 23.11.0 버전 이후로는 기본적으로 pytorch 1.13.1 버전의 도커 이미지를 제공하고 있습니다. 23.11.0 버전 이전으로는 기본적으로 pytorch 1.7.1 버전의 도커 이미지를 제공하고 있습니다.**

<aside>
💡 특정 모레 솔루션 버전에서는 일부 대응하는 pytorch 버전 도커 이미지가 존재하지 않을 수 있습니다. 만일 필요하시다면 문의를 통해서 생성 요청 부탁드리겠습니다.

</aside>

### **Supported Arguments**

- **`pullonly (-p)`**

해당 옵션값을 추가로 줄경우, Moreh 솔루션 이미지를 바로 실행하지 않고 단순히 다운로드하게 됩니다.

해당 옵션값을 사용할 때는 `--target` 옵션값을 추가로 사용할 수 있으며, `--target`옵션 값 뒤에는 아래 예시 명령어와 같이 버전을 명시해줘야 합니다. 만일 없을 경우 최신버전 이미지를 가져오게 됩니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-docker-run --pullonly --target 23.9.0

```

- **`version (-v)`**

Moreh 솔루션 도커 이미지 버전명을 보여줍니다.

```bash
23.9.0

```

- **`—-target {VERSION}`**

특정 Moreh 솔루션 버전의 도커 이미지를 실행합니다. 기본값은 최신 모레 솔루션 버전이 들어가게 됩니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-docker-run --target 23.9.0

```

- **`--torch {VERSION}`**

Moreh 솔루션 도커 이미지내에 설치된 torch 버전을 명시합니다. 기본값은 1.13.1입니다. (***Moreh솔루션 23.11.0 이후***)

```bash
(pytorch) ubuntu@vm:~$ moreh-docker-run --torch 1.13.1

```

- **`--tensorflow {VERSION}`**

<aside>
⚠️ Tensorflow 는 모레 솔루션이 공식 지원을 하지 않는 프레임워크입니다. 따라서, 특정 버전의 이미지가 존재하지 않을 수 있습니다. 필요하신 경우, 문의를 통해서 지원해드리겠습니다.

</aside>

Moreh 솔루션 도커 이미지 내에 설치된 Tensorflow 버전을 명시합니다. 기본값은 2.9.0입니다. 현재 Moreh 솔루션에서는 **tensorflow 2.9.0 버전만 제공 중**인 점 참고부탁드립니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-docker-run --tensorflow 2.9.0
```

## MoAI Platform에서 Docker로 Moreh 솔루션을 실행하는 시나리오

VM에서 다음과 같이 실행하여 AI 가속기가 활성화된 컨테이너를 실행할 수 있습니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-docker-run
...
Login Succeeded
Unable to find image 'sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1' locally
23.9.0-pytorch1.7.1: Pulling from moreh
...
Digest: sha256:b285a30ce74457cc5111b25c7d841f55e9bd090f8ead1a9e79291b7ea03684cc
Status: Downloaded newer image for sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1
(moreh) root@vm:~#
```

만일, 특정 버전의 Moreh 솔루션 이미지를 실행하고 싶다면, 위 명령어 뒤에 `—-target`이라는 옵션을 추가하여 원하시는 Moreh 솔루션 버전 도커 이미지를 실행하실 수 있습니다. 만일 해당 옵션 없이 `moreh-docker-run`을 실행하면 현재까지 배포된 Moreh 솔루션 중 최신 버전으로 이미지를 실행하게 됩니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-docker-run --target 23.9.0
...
<https://docs.docker.com/engine/reference/commandline/login/#credentials-store>
Login Succeeded
Unable to find image 'sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1' locally
23.9.0-pytorch1.7.1: Pulling from moreh
...
Digest: sha256:fce125f57b7e5ede453c0875e93bf7dfa093b77a64418ecfd970ccf69dd94ae4
Status: Downloaded newer image for sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1
(moreh) root@vm:~#
```

컨테이너 안에서 AI 가속기 정보를 조회하고 PyTorch 프로그램을 실행시킬 수 있습니다.

```bash
(moreh) root@vm:~# moreh-smi
04:23:34 September 13, 2023
+------------------------------------------------------------------------------------------------------+
|                                                     Current Version: 23.9.0  Latest Version: 23.9.0  |
+------------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model       |  Memory Usage  |  Total Memory  |  Utilization  |
+======================================================================================================+
|  * 0     |  KT AI Accelerator  |  Small.64GB       |  -             |  -             |  -            |
+------------------------------------------------------------------------------------------------------+
(moreh) root@vm:~# python pytorch-sample.py
Downloading <http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz> to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
26427392it [00:04, 5389702.34it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw
Downloading <http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz> to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
32768it [00:00, 36542.51it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw
Downloading <http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz> to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
4423680it [00:08, 505491.87it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw
Downloading <http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz> to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
8192it [00:00, 13945.86it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
Processing...
Done!
[info] Requesting resources for KT AI Accelerator from the server...
[info] Initializing the worker daemon for KT AI Accelerator...
[info] [1/1] Connecting to resources on the server (192.168.00.00:00000)...
[info] Establishing links to the resources...
[info] KT AI Accelerator is ready to use.
Epoch 1
loss: 2.298501  [    0/60000]
loss: 2.287861  [ 6400/60000]
loss: 2.270298  [12800/60000]

```

컨테이너 안에서 인식되는 AI 가속기는 VM에 할당된 AI 가속기와 동일한 것입니다. VM에서 가속기 모델을 변경하면 컨테이너 안에서도 적용되며 그 반대도 마찬가지입니다. 또한 VM에서 AI 가속기를 사용하는 동안은 컨테이너 안에서는 AI 가속기를 사용할 수 없으며 이것 역시 반대도 마찬가지입니다. 예를 들어 VM에서 AI 가속기를 사용하는 [pytorch-sample.py](http://pytorch-sample.py/) 프로그램이 실행 중인 동안 컨테이너에서 AI 가속기를 사용하는 다른 프로그램을 실행할 경우, 아래와 같은 메시지를 출력하고 VM에서 [pytorch-sample.py](http://pytorch-sample.py/) 프로그램이 끝날 때까지 대기하게 됩니다.

```bash
(moreh) root@vm:~# python pytorch-sample.py
...
[info] Requesting resources for KT AI Accelerator from the server...
[warning] KT AI Accelerator is already in use by another process:
[warning]   (pid: 10000) python pytorch-sample.py
[warning] Two or more processes cannot use KT AI Accelerator at the same time. The program will resume automatically after the process 10000 terminates...

```

이 문서의 나머지 부분에서는 MoAI Platform를 위한 Docker 컨테이너를 실행하는 과정을(즉, `moreh-docker-run` 명령이 내부적으로 하는 일을) 단계별로 자세히 설명합니다.

💡 도커를 사용하지 않고도 VM 안에서 바로 AI 가속기를 사용해 PyTorch 프로그램 실행이 가능합니다. 이 문서는 특별히 도커 기반으로 실행해야 하는 애플리케이션이 있는 분들을 대상으로 합니다.

### 도커 이미지 내려받기

위와 다르게, 단순히 Moreh 솔루션 이미지만 내려받고 싶으시다면 `—-pullonly (-p)` 옵션을 활용하여 이미지를 내려받을수 있습니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-docker-run --pullonly
...
<https://docs.docker.com/engine/reference/commandline/login/#credentials-store>
Login Succeeded
23.9.0-pytorch1.7.1: Pulling from moreh
...
Digest: sha256:b285a30ce74457cc5111b25c7d841f55e9bd090f8ead1a9e79291b7ea03684cc
Status: Downloaded newer image for sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1
sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1

```

해당 명령어도 위와 동일하게 만일 특정 버전의 Moreh 솔루션 이미지를 내려받고싶다면, `—-target` 옵션 추가로 이를 수행하실수가 있습니다. 만일 해당 옵션없이 `moreh-docker-run --pullonly`을 실행하면 현재까지 배포된 Moreh 솔루션중 최신 버전으로 이미지를 실행하게 됩니다.

```bash
(pytorch) ubuntu@vm:~$ moreh-docker-run --pullonly --target 23.9.0
...
<https://docs.docker.com/engine/reference/commandline/login/#credentials-store>
Login Succeeded
23.9.0-pytorch1.7.1: Pulling from moreh
...
Digest: sha256:fce125f57b7e5ede453c0875e93bf7dfa093b77a64418ecfd970ccf69dd94ae4
Status: Downloaded newer image for sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1
sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1

```

### Docker Container runtime으로 컨테이너 시작

`moreh-docker-run` 외에 다음과 같이 `docker run` 명령으로 동일하게 컨테이너를 실행할 수 있습니다. 이 때 다음의 옵션을 포함시켜야 합니다.

- `v /etc/moreh:/etc/moreh`

```bash
(pytorch) ubuntu@vm:~$ docker run --rm -it -v /etc/moreh:/etc/moreh sys.deploy.kt-epc.moreh.io:5001/moreh:23.9.0-pytorch1.7.1 /bin/bash
(moreh) root@vm:~#
```