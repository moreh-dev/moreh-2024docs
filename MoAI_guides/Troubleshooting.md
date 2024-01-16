---
icon: tools
tags: [guide]
order: 10
---


Moreh 솔루션 사용 시 발생할 수 있는 일반적인 오류에 대한 해결 방안을 제공합니다.


### Two or more processes cannot use KT AI Accelerator at the same time.

- SDA가 이미 사용 중인 경우, Two or more processes cannot use KT AI Accelerator at the same time. 경고 메시지가 출력될 수 있습니다. `moreh-smi --reset` 명령을 실행하여 강제로 SDA를 해제할 수 있습니다. 동일한 토큰 값으로 여러 개의 Pod를 띄워 SDA를 동시에 사용하려는 경우(e.g., K8s 기반 서비스) KT Cloud에 문의하여 토큰의 duplicable 설정을 받으시기 바랍니다.
    
    ```bash
    # SDA가 이미 사용 중인 경우의 로그
    
    [2023-09-12 21:47:43.056] [info] Requesting resources for KT AI Accelerator from the server...
    [2023-09-12 21:47:43.070] [warning] KT AI Accelerator is already in use by another process:
    [2023-09-12 21:47:43.070] [warning]   (pid: 3788794) python
    [2023-09-12 21:47:43.070] [warning] Two or more processes cannot use KT AI Accelerator at the same time. The program will resume automatically after the process 3788794 terminates...
    [2023-09-12 21:47:43.070] [warning] If this message is displayed even though the process 3788794 has already terminated, please try to manually release the resources of KT AI Accelerator by executing the "moreh-smi --reset" command 1-2 times.
    ..........
    ```
    
    - `moreh-smi --reset`으로 강제로 SDA 해제
    
    ```bash
    $ moreh-smi --reset
    ```
    

### moreh::InvalidToken.

- SDA 토큰이 적용되지 않아 발생하는 오류 메시지입니다. 환경 변수 `MOREH_SDA_TOKEN`를 할당받은 토큰으로 설정한 후, 모레 솔루션을 사용하시면 해당 오류가 해결됩니다.
    
    ```bash
    $ MOREH_SDA_TOKEN {token} python pytorch_sample.py
    ```
    

### `update-moreh --tensorflow` 명령줄 실행 시 업데이트가 진행되지 않거나 Python 패키지가 잡히지 않는 경우.

- 해당 문제는 `.local` 폴더와 관련된 문제일 수 있습니다. 해당 폴더 `~/.local/lib`과 `~/.local/bin`을 삭제 후 재시도해보시기 바랍니다.

### 사용자 VM에서 Python 프로세스가 종료되지 않고 남아있는 경우

- 모델 학습을 강제 중단 혹은 종료한다면 비정상 종료된 Python 프로세스가 종료되지 않고 남아있을 수 있습니다.
- `pkill python` 혹은 `vkill {pid}` 를 통해 종료하시길 바랍니다.

### SSH 클라이언트와 통신이 끊겨 학습이 종료되는 경우

- 보안을 위해 일정 시간 터미널에서 동작이 없다면 SSH 클라이언트와 통신이 끊기게 됩니다.
- 위와 같이 학습이 종료되는 문제를 방지하기 위하여, tmux 등의 터미널 다중화 프로그램을 이용하시는 것을 권장드립니다.

### 23.8.0 이전 버전을 update-moreh 명령어로 설치할 때 Moreh 솔루션이 제대로 설치되지 않는 경우

- `update-moreh` 명령어 외에 파이썬 패키지 관리자인 pip을 사용하여 Moreh 솔루션을 동일하게 업데이트할 수 있습니다.
- `update-moreh`로 솔루션 설치가 제대로 이루어지지 않는다면 다음의 `pip install`을 통해 솔루션을 설치해보시기 바랍니다.

```bash
# pytorch 1.7.1
$ pip install torch==1.7.1+cu110.moreh23.9.0

# pytorch 1.10.0
$ pip install torch==1.10.0+cu111.moreh23.9.0

# pytorch 1.13.1
$ pip install torch==1.13.1+cu116.moreh23.9.0

# tensorflow 2.9.0 (pytorch 1.7.1+moreh 솔루션 패키지를 필요로 합니다.)
$ pip install torch==1.7.1+cu110.moreh23.9.0
$ pip install tensorflow==2.9.0+moreh23.9.0
```

### Moreh 솔루션 버전을 roll back하려 할 때 update-moreh가 제대로 동작하지 않는 경우

- 동일한 버전으로 다시 Moreh 솔루션을 설치하고 싶은 경우나, 원하는 타깃 버전으로 업(다운)그레이드가 정상적으로 되지 않을 경우 `-force` 옵션을 통해 강제로 Moreh 솔루션의 업데이트를 진행할 수 있습니다.
- 하지만 가급적이면 Moreh 솔루션 사용시 `-force` command는 지양하고 있으니 버전이슈로 인한 에러가 발생한 경우 고객지원을 요청 바랍니다.

```bash
# 23.9.0 버전을 강제 설치
update-moreh --target 23.9.0 --force
```