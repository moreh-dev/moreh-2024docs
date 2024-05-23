---
icon: terminal
tags:  [tutorial, gpt]
order: 40
---

# 5. Changing the Number of GPUs

Let's rerun the fine-tuning task with a different number of GPUs. MoAI Platform abstracts GPU resources into a single accelerator and automatically performs parallel processing. Therefore, there is no need to modify the PyTorch script even when changing the number of GPUs.

## Changing the Accelerator type

Switch the accelerator type using the **`moreh-switch-model`** tool. For instructions on changing the accelerator, please refer again to the [**3. Model fine-tuning**](3_finetuning.md) document.

```bash
$ moreh-switch-model
```

Please contact your infrastructure provider and choose one of the following options before proceeding.   ([**KT Hyperscale AI Computing (HAC) AI Accelerator Information**](/Supported_Documents/KT_HAC_Models_Info.md))

- AMD MI250 GPU with 32 units
    - When using Moreh's trial container: select [!badge variant="secondary" text="8xlarge"]
    - When using KT Cloud's Hyperscale AI Computing: select [!badge variant="secondary" text="8xLarge.4096GB"]
- AMD MI210 GPU with 64 units
- AMD MI300X GPU with 16 units

## Training Parameters

Run the **`train_gpt.py`** script again.

```bash
~/moreh-quickstart$ python tutorial/train_gpt.py --batch-size 32
```

Since the available GPU memory has doubled, let's also change the batch size to 32 and run it.

If the training proceeds normally, the following log will be output.


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

Compared to the previous execution results when the number of GPUs was half, you can see that the learning is the same and the throughput has improved. 

- When using AMD MI250 GPU 16 → 32 : approximately 6800 tokens/sec → 13000 tokens/sec