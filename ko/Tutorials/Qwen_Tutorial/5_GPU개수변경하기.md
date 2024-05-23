---
icon: terminal
tags: [guide]
order: 40
---

# 5. GPU 개수 변경하기

앞과 동일한 fine tuning 작업을 GPU 개수를 바꾸어 다시 실행해 보겠습니다. MoAI Platform은 GPU 자원을 단일 가속기로 추상화하여 제공하며 자동으로 병렬 처리를 수행합니다. 그러므로 GPU 개수를 변경하더라도 PyTorch 스크립트를 수정할 필요가 전혀 없습니다.

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

다시 별도의 배치 사이즈 변경 없이 `train_qwen.py` 스크립트를 실행합니다.

```bash
~/quickstart$ python tutorial/train_qwen.py
```

학습이 정상적으로 진행된다면 다음과 같은 로그가 출력될 것입니다.

```bash
2024-04-19 03:07:02,942 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmp7rrxrdcb
2024-04-19 03:07:02,943 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmp7rrxrdcb/_remote_module_non_scriptable.py
Downloading shards: 100%|██████████| 4/4 [00:00<00:00, 14425.81it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:06<00:00,  1.54s/it][2024-04-19 03:07:40.492] [info] Got DBs from backend for auto config.
[2024-04-19 03:07:42.126] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-19 03:07:42.136] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-19 03:07:47.316] [info] [1/8] Connecting to resources on the server (192.168.110.10:24155)...
[2024-04-19 03:07:47.328] [info] [2/8] Connecting to resources on the server (192.168.110.12:24155)...
[2024-04-19 03:07:47.336] [info] [3/8] Connecting to resources on the server (192.168.110.26:24155)...
[2024-04-19 03:07:47.342] [info] [4/8] Connecting to resources on the server (192.168.110.32:24155)...
[2024-04-19 03:07:47.349] [info] [5/8] Connecting to resources on the server (192.168.110.51:24155)...
[2024-04-19 03:07:47.356] [info] [6/8] Connecting to resources on the server (192.168.110.78:24155)...
[2024-04-19 03:07:47.362] [info] [7/8] Connecting to resources on the server (192.168.110.96:24155)...
[2024-04-19 03:07:47.369] [info] [8/8] Connecting to resources on the server (192.168.110.97:24155)...
[2024-04-19 03:07:47.378] [info] Establishing links to the resources...
[2024-04-19 03:07:48.249] [info] MoAI Accelerator is ready to use.
[2024-04-19 03:07:48.619] [info] The number of candidates is 22.
[2024-04-19 03:07:48.619] [info] Parallel Graph Compile start...
[2024-04-19 03:08:17.031] [info] Elapsed Time to compile all candidates = 28411 [ms]
[2024-04-19 03:08:17.031] [info] Parallel Graph Compile finished.
[2024-04-19 03:08:17.031] [info] The number of possible candidates is 4.
[2024-04-19 03:08:17.031] [info] SelectBestGraphFromCandidates start...
[2024-04-19 03:08:17.693] [info] Elapsed Time to compute cost for survived candidates = 661 [ms]
[2024-04-19 03:08:17.693] [info] SelectBestGraphFromCandidates finished.
[2024-04-19 03:08:17.693] [info] Configuration for parallelism is selected.
[2024-04-19 03:08:17.693] [info] num_stages : 2, num_micro_batches : 8, batch_per_device : 1, No TP, recomputation : false, distribute_param : true
[2024-04-19 03:08:17.694] [info] train: true

2024-04-19 03:09:13.571 | INFO     | __main__:main:154 - [Step 1/144] | Loss: 1.03125 | Duration: 39.51 | Throughput: 13270.26 tokens/sec
2024-04-19 03:09:23.458 | INFO     | __main__:main:154 - [Step 2/144] | Loss: 1.0859375 | Duration: 4.37 | Throughput: 119959.56 tokens/sec
2024-04-19 03:09:33.195 | INFO     | __main__:main:154 - [Step 3/144] | Loss: 0.8984375 | Duration: 4.44 | Throughput: 118024.31 tokens/sec
2024-04-19 03:09:43.861 | INFO     | __main__:main:154 - [Step 4/144] | Loss: 0.85546875 | Duration: 5.30 | Throughput: 99006.09 tokens/sec
2024-04-19 03:09:54.854 | INFO     | __main__:main:154 - [Step 5/144] | Loss: 0.890625 | Duration: 5.72 | Throughput: 91618.65 tokens/sec
...
2024-04-19 03:33:00.158 | INFO     | __main__:main:154 - [Step 141/144] | Loss: 0.46875 | Duration: 5.28 | Throughput: 99212.01 tokens/sec
2024-04-19 03:33:09.622 | INFO     | __main__:main:154 - [Step 142/144] | Loss: 0.45703125 | Duration: 4.35 | Throughput: 120536.34 tokens/sec
2024-04-19 03:33:19.308 | INFO     | __main__:main:154 - [Step 143/144] | Loss: 0.451171875 | Duration: 4.35 | Throughput: 120554.52 tokens/sec
2024-04-19 03:33:28.985 | INFO     | __main__:main:154 - [Step 144/144] | Loss: 0.443359375 | Duration: 4.41 | Throughput: 118957.15 tokens/sec
...
```

앞서 GPU 개수가 절반이었을 때 실행한 결과와 비교해 동일하게 학습이 이루어지며 throughput이 향상되었음을 확인할 수 있습니다.

- AMD MI250 GPU 16 → 32개 사용 시: 약 59,000 tokens/sec → 105,000 tokens/sec