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

다시 `train_llama2.py` 스크립트를 실행합니다.

```bash
~/quickstart$ python tutorial/train_llama2.py --batch-size 512
```

사용 가능한 GPU 메모리가 **2배** 늘었기 때문에, 배치 사이즈 또한 기존 `256` 에서 `512`로 변경하여 실행시켜 보겠습니다. 

학습이 정상적으로 진행된다면 다음과 같은 로그가 출력될 것입니다.

```bash
...
[2024-04-25 15:56:09.831] [info] Requesting resources for KT AI Accelerator from the server...
[2024-04-25 15:56:09.841] [info] Initializing the worker daemon for KT AI Accelerator
[2024-04-25 15:56:15.130] [info] [1/8] Connecting to resources on the server (192.168.110.1:24166)...
[2024-04-25 15:56:15.143] [info] [2/8] Connecting to resources on the server (192.168.110.24:24166)...
[2024-04-25 15:56:15.150] [info] [3/8] Connecting to resources on the server (192.168.110.25:24166)...
[2024-04-25 15:56:15.156] [info] [4/8] Connecting to resources on the server (192.168.110.26:24166)...
[2024-04-25 15:56:15.164] [info] [5/8] Connecting to resources on the server (192.168.110.51:24166)...
[2024-04-25 15:56:15.170] [info] [6/8] Connecting to resources on the server (192.168.110.79:24166)...
[2024-04-25 15:56:15.177] [info] [7/8] Connecting to resources on the server (192.168.110.80:24166)...
[2024-04-25 15:56:15.184] [info] [8/8] Connecting to resources on the server (192.168.110.99:24166)...
[2024-04-25 15:56:15.191] [info] Establishing links to the resources...
[2024-04-25 15:56:16.061] [info] KT AI Accelerator is ready to use.
[2024-04-25 15:56:16.521] [info] The number of candidates is 24.
[2024-04-25 15:56:16.521] [info] Parallel Graph Compile start...
[2024-04-25 15:57:14.441] [info] Elapsed Time to compile all candidates = 57920 [ms]
[2024-04-25 15:57:14.441] [info] Parallel Graph Compile finished.
[2024-04-25 15:57:14.441] [info] The number of possible candidates is 3.
[2024-04-25 15:57:14.441] [info] SelectBestGraphFromCandidates start...
[2024-04-25 15:57:16.562] [info] Elapsed Time to compute cost for survived candidates = 2120 [ms]
[2024-04-25 15:57:16.562] [info] SelectBestGraphFromCandidates finished.
[2024-04-25 15:57:16.562] [info] Configuration for parallelism is selected.
[2024-04-25 15:57:16.562] [info] num_stages : 2, num_micro_batches : 16, batch_per_device : 1, No TP, recomputation : true, distribute_param : true
[2024-04-25 15:57:16.562] [info] train: true

2024-04-25 15:59:00.492 | INFO     | __main__:main:136 - [Step 2/560] | Loss: 1.6953125 | Duration: 15.12 | Throughput: 69337.98 tokens/sec
2024-04-25 16:00:05.283 | INFO     | __main__:main:136 - [Step 4/560] | Loss: 1.6875 | Duration: 15.93 | Throughput: 65842.99 tokens/sec
2024-04-25 16:01:12.262 | INFO     | __main__:main:136 - [Step 6/560] | Loss: 1.703125 | Duration: 15.73 | Throughput: 66656.28 tokens/sec
2024-04-25 16:02:19.591 | INFO     | __main__:main:136 - [Step 8/560] | Loss: 1.6328125 | Duration: 15.36 | Throughput: 68263.53 tokens/sec
2024-04-25 16:03:24.498 | INFO     | __main__:main:136 - [Step 10/560] | Loss: 1.5859375 | Duration: 12.78 | Throughput: 82040.81 tokens/sec
2024-04-25 16:04:28.820 | INFO     | __main__:main:136 - [Step 12/560] | Loss: 1.59375 | Duration: 13.00 | Throughput: 80657.85 tokens/sec
2024-04-25 16:05:32.933 | INFO     | __main__:main:136 - [Step 14/560] | Loss: 1.6328125 | Duration: 12.65 | Throughput: 82906.48 tokens/sec
2024-04-25 16:06:37.614 | INFO     | __main__:main:136 - [Step 16/560] | Loss: 1.6796875 | Duration: 14.94 | Throughput: 70195.58 tokens/sec
2024-04-25 16:07:44.641 | INFO     | __main__:main:136 - [Step 18/560] | Loss: 1.6875 | Duration: 13.01 | Throughput: 80607.34 tokens/sec
...
```

앞서 GPU 개수가 절반이었을 때 실행한 결과와 비교해 동일하게 학습이 이루어지며 throughput이 향상되었음을 확인할 수 있습니다.

- AMD MI250 GPU 16 → 32개 사용 시: 약 35,000 tokens/sec → 74,000 tokens/sec