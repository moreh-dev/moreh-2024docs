---
icon: terminal
tags: [guide]
order: 40
---

# 5. GPU 개수 변경하기

앞과 동일한 fine-tuning 작업을 GPU 개수를 바꾸어 다시 실행해 보겠습니다. MoAI Platform은 GPU 자원을 단일 가속기로 추상화하여 제공하며 자동으로 병렬 처리를 수행합니다. 따라서 GPU 개수를 변경하더라도 PyTorch 스크립트를 수정할 필요가 전혀 없습니다.

## 가속기 Flavor 변경

`moreh-switch-model` 툴을 사용하여 가속기 flavor를 전환합니다. 가속기 변경 방법은 [3. 학습 실행하기](3_학습_실행하기.md) 문서를 한번 더 참고해주시기 바랍니다.

```bash
$ moreh-switch-model
```

인프라 제공자에게 문의하여 다음 중 하나를 선택한 다음 계속 진행하십시오.  ([KT Hyperscale AI Computing (HAC) 서비스 가속기 모델 정보](/Supported_Documents/KT_HAC_Models_Info.md))


- AMD MI250 GPU 32개 사용
    - Moreh의 체험판 컨테이너 사용 시: [!badge variant="secondary" text="8xlarge"] 선택
    - KT Cloud의 Hyperscale AI Computing 사용 시: [!badge variant="secondary" text="8xLarge.4096GB"] 선택
- AMD MI210 GPU 64개 사용
- AMD MI300X GPU 16개 사용

## 학습 실행

다시 `train_llama3.py` 스크립트를 실행합니다.

```bash
~/quickstart$ python tutorial/train_llama3.py --batch-size 512
```

사용 가능한 GPU 메모리가 **2배** 늘었기 때문에, 배치 사이즈 또한 기존 `256` 에서 `512`로 변경하여 실행시켜 보겠습니다. 

학습이 정상적으로 진행된다면 다음과 같은 로그가 출력될 것입니다.

```bash
...
[2024-05-13 18:04:23.681] [info] Got DBs from backend for auto config.
[2024-05-13 18:04:25.833] [info] Requesting resources for KT AI Accelerator from the server...
[2024-05-13 18:04:25.844] [info] Initializing the worker daemon for KT AI Accelerator
[2024-05-13 18:04:30.622] [info] [1/8] Connecting to resources on the server (192.168.110.4:24172)...
[2024-05-13 18:04:30.637] [info] [2/8] Connecting to resources on the server (192.168.110.5:24172)...
[2024-05-13 18:04:30.645] [info] [3/8] Connecting to resources on the server (192.168.110.10:24172)...
[2024-05-13 18:04:30.651] [info] [4/8] Connecting to resources on the server (192.168.110.42:24172)...
[2024-05-13 18:04:30.658] [info] [5/8] Connecting to resources on the server (192.168.110.43:24172)...
[2024-05-13 18:04:30.665] [info] [6/8] Connecting to resources on the server (192.168.110.44:24172)...
[2024-05-13 18:04:30.672] [info] [7/8] Connecting to resources on the server (192.168.110.83:24172)...
[2024-05-13 18:04:30.679] [info] [8/8] Connecting to resources on the server (192.168.110.84:24172)...
[2024-05-13 18:04:30.686] [info] Establishing links to the resources...
[2024-05-13 18:04:31.612] [info] KT AI Accelerator is ready to use.
[2024-05-13 18:04:31.612] [info] Moreh Version: 24.5.0
[2024-05-13 18:04:31.612] [info] Moreh Job ID: 976907
[2024-05-13 18:04:31.835] [warning] Various batch size detected : 512, 1
[2024-05-13 18:04:31.835] [info] The number of candidates is 6.
[2024-05-13 18:04:31.835] [info] Parallel Graph Compile start...
[2024-05-13 18:04:32.468] [info] Elapsed Time to compile all candidates = 633 [ms]
[2024-05-13 18:04:32.469] [info] Parallel Graph Compile finished.
[2024-05-13 18:04:32.469] [info] The number of possible candidates is 2.
[2024-05-13 18:04:32.469] [info] SelectBestGraphFromCandidates start...
[2024-05-13 18:04:32.707] [info] Elapsed Time to compute cost for survived candidates = 238 [ms]
[2024-05-13 18:04:32.708] [info] SelectBestGraphFromCandidates finished.
[2024-05-13 18:04:32.708] [info] Configuration for parallelism is selected.
[2024-05-13 18:04:32.708] [info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : false
[2024-05-13 18:04:32.708] [info] train: true
2024-05-13 18:08:43.443 | INFO     | main:main:135 - [Step 2/560] | Loss: 2.1875 | Duration: 1.89 | Throughput: 276803.38 tokens/sec
2024-05-13 18:09:01.651 | INFO     | main:main:135 - [Step 4/560] | Loss: 2.109375 | Duration: 1.40 | Throughput: 375362.04 tokens/sec
2024-05-13 18:09:19.639 | INFO     | main:main:135 - [Step 6/560] | Loss: 2.046875 | Duration: 1.16 | Throughput: 450234.51 tokens/sec
2024-05-13 18:09:37.844 | INFO     | main:main:135 - [Step 8/560] | Loss: 2.015625 | Duration: 1.35 | Throughput: 387487.33 tokens/sec
2024-05-13 18:09:55.952 | INFO     | main:main:135 - [Step 10/560] | Loss: 2.015625 | Duration: 1.33 | Throughput: 393661.22 tokens/sec
...
```

앞서 GPU 개수가 절반이었을 때 실행한 결과와 비교해 동일하게 학습이 이루어지며 throughput이 향상되었음을 확인할 수 있습니다.

- AMD MI250 GPU 16 → 32개 사용 시: 약 200,000 tokens/sec → 390,000 tokens/sec