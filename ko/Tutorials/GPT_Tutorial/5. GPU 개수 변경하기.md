---
icon: terminal
tags: [guide]
order: 40
---

# 5. GPU 개수 변경하기

앞과 동일한 fine-tuning 작업을 GPU 개수를 바꾸어 다시 실행해 보겠습니다. MoAI Platform은 GPU 자원을 단일 가속기로 추상화하여 제공하며 자동으로 병렬 처리를 수행합니다. 따라서 GPU 개수를 변경하더라도 PyTorch 스크립트는 전혀 고칠 필요가 없습니다. 

## 가속기 Flavor 변경

`moreh-switch-model` 툴을 사용하여 가속기 flavor를 전환합니다. 가속기 변경 방법은 [3. 학습 실행하기](3_학습_실행하기.md) 문서를 한번 더 참고해주시기 바랍니다.

```
$ moreh-switch-model
```

인프라 제공자에게 문의하여 다음 중 하나를 선택한 다음 계속 진행하십시오.  ([KT Hyperscale AI Computing (HAC) 서비스 가속기 모델 정보](/Supported_Documents/KT_HAC_Models_Info.md))

- AMD MI250 GPU 32개 사용
    - Moreh의 체험판 컨테이너 사용 시: [!badge variant="secondary" text="8xlarge"] 선택
    - KT Cloud의 Hyperscale AI Computing 사용 시: [!badge variant="secondary" text="8xLarge.4096GB"] 선택
- AMD MI210 GPU 64개 사용
- AMD MI300X GPU 16개 사용

## 학습 실행

다시 `train_gpt.py` 스크립트를 실행합니다.

```bash
~/moreh-quickstart$ python tutorial/train_gpt.py --batch-size 64
```

사용 가능한 GPU 메모리가 2배가 늘었기 때문에, 배치 사이즈 또한 64로 변경하여 실행시켜 보겠습니다. 

학습이 정상적으로 진행된다면 다음과 같은 로그가 출력될 것입니다.

```bash
2024-04-19 18:19:08,362 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpfd9q7p9n
2024-04-19 18:19:08,362 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpfd9q7p9n/_remote_module_non_scriptable.py
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████| 2/2 [01:02<00:00, 31.27s/it]
2024-04-19 18:20:48,957 - numexpr.utils - INFO - Note: NumExpr detected 16 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
2024-04-19 18:20:48,957 - numexpr.utils - INFO - NumExpr defaulting to 8 threads.
2024-04-19 18:20:49,228 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
2024-04-19 18:20:49,229 - datasets - INFO - Apache Beam version 2.46.0 available.
[2024-04-19 18:20:50.025] [info] Got DBs from backend for auto config.
[2024-04-19 18:20:53.197] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-19 18:20:53.210] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-19 18:20:58.513] [info] [1/8] Connecting to resources on the server (192.168.110.19:24162)...
[2024-04-19 18:20:58.527] [info] [2/8] Connecting to resources on the server (192.168.110.20:24162)...
[2024-04-19 18:20:58.534] [info] [3/8] Connecting to resources on the server (192.168.110.42:24162)...
[2024-04-19 18:20:58.543] [info] [4/8] Connecting to resources on the server (192.168.110.43:24162)...
[2024-04-19 18:20:58.551] [info] [5/8] Connecting to resources on the server (192.168.110.72:24162)...
[2024-04-19 18:20:58.566] [info] [6/8] Connecting to resources on the server (192.168.110.73:24162)...
[2024-04-19 18:20:58.576] [info] [7/8] Connecting to resources on the server (192.168.110.91:24162)...
[2024-04-19 18:20:58.588] [info] [8/8] Connecting to resources on the server (192.168.110.93:24162)...
[2024-04-19 18:20:58.596] [info] Establishing links to the resources...
[2024-04-19 18:20:59.482] [info] MoAI Accelerator is ready to use.
[2024-04-19 18:20:59.843] [info] The number of candidates is 12.
[2024-04-19 18:20:59.843] [info] Parallel Graph Compile start...
[2024-04-19 18:21:10.762] [info] Elapsed Time to compile all candidates = 10919 [ms]
[2024-04-19 18:21:10.762] [info] Parallel Graph Compile finished.
[2024-04-19 18:21:10.762] [info] The number of possible candidates is 3.
[2024-04-19 18:21:10.762] [info] SelectBestGraphFromCandidates start...
[2024-04-19 18:21:11.247] [info] Elapsed Time to compute cost for survived candidates = 484 [ms]
[2024-04-19 18:21:11.247] [info] SelectBestGraphFromCandidates finished.
[2024-04-19 18:21:11.247] [info] Configuration for parallelism is selected.
[2024-04-19 18:21:11.247] [info] num_stages : 4, num_micro_batches : 4, batch_per_device : 1, No TP, recomputation : true, distribute_param : true
[2024-04-19 18:21:11.248] [info] train: true
2024-04-19 18:23:14.630 | INFO     | __main__:main:81 - [Step 1/1660] Loss: 0.9296875 Throughput: 5668.73 tokens/sec
2024-04-19 18:23:24.785 | INFO     | __main__:main:81 - [Step 2/1660] Loss: 0.7734375 Throughput: 34748.50 tokens/sec
```

앞서 GPU 개수가 절반이었을 때 실행한 결과와 비교해 동일하게 학습이 이루어지며 throughput이 향상되었음을 확인할 수 있습니다.

- AMD MI250 GPU 16 → 32개 사용 시: 약 17000 tokens/sec → 34000 tokens/sec