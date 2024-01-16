---
order: 100
icon: rocket
tags: [guide]
---
# Getting Started

이 매뉴얼은 개발자가 터미널에 접속해서 MoAI 플랫폼을 이용하는 Quickstart 가이드를 제공합니다. 


## 서버 접속하기

관리자가 이용자에게 제공한 VM 접속정보(IP, Port, SSH Key)를 입력해 접속할 수 있습니다.

여기서는 아래 VM 접속정보를 제공하였다고 가정하겠습니다.

```yaml
hostname: vm
ip: xxx.00.xx.000
port: 22
user: ubuntu
permission key: key.pem
```

다음은 VM 접속정보를 활용하여 해당 VM에 ssh 명령어로 접속하는 예시입니다.

```bash
$ ssh -i key.pem -p 22 ubuntu@xxx.00.xx.000
Welcome to Ubuntu 20.04.4 LTS (GNU/Linux 5.13.0-35-generic x86_64)

* Documentation:  https://help.ubuntu.com
* Management:     https://landscape.canonical.com
* Support:        https://ubuntu.com/advantage

    (...)

(moreh) ubuntu@vm:~$
```

## 모델 학습 실행하기

여기서는 Bert 모델을 학습해 보겠습니다. 먼저 Bert 모델을 다운로드합니다.

```bash
ubuntu@vm:~$ get-reference-model bert
```

설치 완료 이후 학습하고자 하는 모델 폴더로 이동합니다.

```bash
# 현재 디렉토리 목록 확인
ubuntu@vm:~$ ls
anaconda3  morehtools  resnet   bert   sample

# 학습 모델 폴더로 이동
ubuntu@vm:~$ cd bert

# conda 환경 활성화
ubuntu@vm:~/bert$ conda activate pytorch

# Bert 모델 사전 학습 시작
(pytorch) ubuntu@vm:~/bert$ python train.py
```

- `python train.py` 를 실행하여 Bert 모델의 사전 학습을 진행합니다.
- 모델 학습을 강제 중단 혹은 종료로 인해 Python 프로세스가 종료되지 않는 경우를 방지하기 위해, `moreh-smi -r` 명령어를 통해 학습 프로세스를 종료하시길 권장드립니다.
- SSH 클라이언트와 통신이 끊겨 학습이 종료되는 문제를 방지하기 위하여, tmux 등의 터미널 다중화 프로그램을 이용하시는 것을 권장드립니다.

`moreh-smi`를 통해 실행 중인 학습 프로세스 및 GPU 자원 사용량을 확인할 수 있습니다.

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