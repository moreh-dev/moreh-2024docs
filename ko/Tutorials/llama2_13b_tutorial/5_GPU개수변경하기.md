---
icon: terminal
tags: [tutorial, llama2]
order: 40
---

# 5. GPU 개수 변경하기

앞과 동일한 fine-tuning 작업을 GPU 개수를 바꾸어 다시 실행해 보겠습니다. MoAI Platform은 GPU 자원을 단일 가속기로 추상화하여 제공하며 자동으로 병렬 처리를 수행합니다. 따라서 GPU 개수를 변경하더라도 PyTorch 스크립트를 수정할 필요가 전혀 없습니다.

## 가속기 Flavor 변경

`moreh-switch-model` 툴을 사용하여 가속기 flavor를 전환합니다. 가속기 변경 방법은 [3. 학습 실행하기](3_학습_실행하기.md) 문서를 한번 더 참고해주시기 바랍니다.

```bash
$ moreh-switch-model
```

- AMD MI250 GPU 32개 사용
    - Moreh의 체험판 컨테이너 사용 시: [!badge variant="secondary" text="8xlarge"] 선택
    - KT Cloud의 Hyperscale AI Computing 사용 시: [!badge variant="secondary" text="8xLarge.4096GB"] 선택
- AMD MI210 GPU 64개 사용
- AMD MI300X GPU 16개 사용

## 학습 실행

다시 `train_llama2.py` 스크립트를 실행합니다.

```bash
~/quickstart$ python tutorial/train_llama2.py --batch-size 512
```

사용 가능한 GPU 메모리가 **2배** 늘었기 때문에, 배치 사이즈 또한 기존 `256` 에서 `512`로 변경하여 실행시켜 보겠습니다. 

학습이 정상적으로 진행된다면 다음과 같은 로그가 출력될 것입니다.

```bash
...
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.12:24162)...
[info] [2/8] Connecting to resources on the server (192.168.110.13:24162)...
[info] [3/8] Connecting to resources on the server (192.168.110.14:24162)...
[info] [4/8] Connecting to resources on the server (192.168.110.37:24162)...
[info] [5/8] Connecting to resources on the server (192.168.110.39:24162)...
[info] [6/8] Connecting to resources on the server (192.168.110.73:24162)...
[info] [7/8] Connecting to resources on the server (192.168.110.75:24162)...
[info] [8/8] Connecting to resources on the server (192.168.110.99:24162)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 977792
[info] The number of candidates is 78.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 190703 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 58.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 110073 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : true
[info] train: true

| INFO     | __main__:main:136 - [Step 2/560] | Loss: 1.8515625 | Duration: 3.62 | Throughput: 144909.28 tokens/sec
| INFO     | __main__:main:136 - [Step 4/560] | Loss: 1.765625 | Duration: 1.59 | Throughput: 328842.71 tokens/sec
| INFO     | __main__:main:136 - [Step 6/560] | Loss: 1.734375 | Duration: 1.51 | Throughput: 346245.79 tokens/sec
| INFO     | __main__:main:136 - [Step 8/560] | Loss: 1.703125 | Duration: 1.70 | Throughput: 307668.92 tokens/sec
...
```

앞서 GPU 개수가 절반이었을 때 실행한 결과와 비교해 동일하게 학습이 이루어지며 throughput이 향상되었음을 확인할 수 있습니다.

- AMD MI250 GPU 16 → 32개 사용 시: 약 150,000 tokens/sec → 315,000 tokens/sec
