---
icon: terminal
tags: [guide]
order: 40
---

# 5. GPU 개수 변경하기

앞과 동일한 fine-tuning 작업을 GPU 개수를 바꾸어 다시 실행해 보겠습니다. GPU 개수를 변경하여 다시 실행해 보겠습니다. MoAI Platform은 GPU 자원을 단일 가속기로 추상화하여 제공하므로 자동으로 병렬 처리를 수행합니다. 따라서 GPU 개수를 변경하더라도 PyTorch 스크립트를 전혀 수정할 필요가 없습니다.

## 가속기 Flavor 변경

`moreh-switch-model` 을 사용하여 가속기 flavor를 전환합니다. 가속기 변경 방법은 [3. 학습 실행하기](3_학습_실행하기.md) 문서를 한번 더 참고해주시기 바랍니다.

```
$ moreh-switch-model
```

인프라 제공자에게 문의하여 다음 중 하나를 선택한 다음 계속 진행하십시오. ([KT Hyperscale AI Computing (HAC) 서비스 가속기 모델 정보](/Supported_Documents/KT_HAC_Models_Info.md))

- AMD MI250 GPU 32개 사용
    - Moreh의 체험판 컨테이너 사용 시: [!badge variant="secondary" text=8xlarge]  선택
    - KT Cloud의 Hyperscale AI Computing 사용 시: [!badge variant="secondary" text=8xLarge.4096GB] 선택
- AMD MI210 GPU 64개 사용
- AMD MI300X GPU 16개 사용



## 학습 실행

배치 사이즈 변경 없이  `train_mistral.py` 스크립트를 실행합니다.

```bash
~/moreh-quickstart$ python tutorial/train_mistral.py
```

학습이 정상적으로 진행된다면 다음과 같은 로그가 출력될 것입니다.

```bash
2024-04-22 02:23:38,069 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpomm9oi6e
2024-04-22 02:23:38,069 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpomm9oi6e/_remote_module_non_scriptable.py
Downloading shards: 100%|██████████| 2/2 [00:00<00:00, 11586.48it/s]
Loading checkpoint shards: 100%|██████████| 2/2 [00:05<00:00,  2.73s/it]
[2024-04-22 02:24:13.852] [info] Got DBs from backend for auto config.
[2024-04-22 02:24:16.623] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-22 02:24:16.635] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-22 02:24:22.012] [info] [1/8] Connecting to resources on the server (192.168.110.10:24174)...
[2024-04-22 02:24:22.025] [info] [2/8] Connecting to resources on the server (192.168.110.33:24174)...
[2024-04-22 02:24:22.032] [info] [3/8] Connecting to resources on the server (192.168.110.34:24174)...
[2024-04-22 02:24:22.039] [info] [4/8] Connecting to resources on the server (192.168.110.52:24174)...
[2024-04-22 02:24:22.045] [info] [5/8] Connecting to resources on the server (192.168.110.53:24174)...
[2024-04-22 02:24:22.054] [info] [6/8] Connecting to resources on the server (192.168.110.79:24174)...
[2024-04-22 02:24:22.063] [info] [7/8] Connecting to resources on the server (192.168.110.80:24174)...
[2024-04-22 02:24:22.073] [info] [8/8] Connecting to resources on the server (192.168.110.98:24174)...
[2024-04-22 02:24:22.081] [info] Establishing links to the resources...
[2024-04-22 02:24:22.940] [info] MoAI Accelerator is ready to use.
[2024-04-22 02:24:23.307] [info] The number of candidates is 22.
[2024-04-22 02:24:23.307] [info] Parallel Graph Compile start...
[2024-04-22 02:24:51.810] [info] Elapsed Time to compile all candidates = 28502 [ms]
[2024-04-22 02:24:51.810] [info] Parallel Graph Compile finished.
[2024-04-22 02:24:51.810] [info] The number of possible candidates is 4.
[2024-04-22 02:24:51.810] [info] SelectBestGraphFromCandidates start...
[2024-04-22 02:24:52.800] [info] Elapsed Time to compute cost for survived candidates = 990 [ms]
[2024-04-22 02:24:52.800] [info] SelectBestGraphFromCandidates finished.
[2024-04-22 02:24:52.800] [info] Configuration for parallelism is selected.
[2024-04-22 02:24:52.800] [info] num_stages : 2, num_micro_batches : 8, batch_per_device : 1, No TP, recomputation : false, distribute_param : true
[2024-04-22 02:24:52.803] [info] train: true

2024-04-22 02:25:47.612 | INFO     | __main__:main:137 - [Step 1/144] | Loss: 1.1953125 | Duration: 41.70 | Throughput: 12572.33 tokens/sec
2024-04-22 02:26:00.323 | INFO     | __main__:main:137 - [Step 2/144] | Loss: 0.85546875 | Duration: 4.68 | Throughput: 111965.54 tokens/sec
2024-04-22 02:26:13.809 | INFO     | __main__:main:137 - [Step 3/144] | Loss: 0.796875 | Duration: 5.81 | Throughput: 90209.43 tokens/sec
2024-04-22 02:26:27.255 | INFO     | __main__:main:137 - [Step 4/144] | Loss: 0.75390625 | Duration: 5.80 | Throughput: 90425.87 tokens/sec
2024-04-22 02:26:39.413 | INFO     | __main__:main:137 - [Step 5/144] | Loss: 0.64453125 | Duration: 4.38 | Throughput: 119712.50 tokens/sec
...

```

앞서 GPU 개수가 절반이었을 때 실행한 결과와 비교해 동일하게 학습이 이루어지며 throughput이 향상되었음을 확인할 수 있습니다.

- AMD MI250 GPU 16 → 32개 사용 시: 약 60,000 tokens/sec → 110,000 tokens/sec