---
icon: terminal
tags: [tutorial, qwen]
order: 40
---

# 5. Changing the Number of GPUs

Let's rerun the fine-tuning task with a different number of GPUs. MoAI Platform abstracts GPU resources into a single accelerator and automatically performs parallel processing. Therefore, there is no need to modify the PyTorch script even when changing the number of GPUs.

## Changing the Accelerator type

Switch the accelerator type using the `moreh-switch-model` tool. For instructions on changing the accelerator, please refer to the [**3. Model fine-tuning**](3_fine_tuning.md) 

```bash
$ moreh-switch-model
```

Please contact your infrastructure provider and choose one of the following options before proceeding.   
[**KT Hyperscale AI Computing (HAC) AI Accelerator Information**](/Supported_Documents/KT_HAC_Models_Info.md)

- AMD MI250 GPU with 32 units
    - When using Moreh's trial container: select [!badge variant="secondary" text="8xlarge"]
    - When using KT Cloud's Hyperscale AI Computing: select [!badge variant="secondary" text="8xLarge.4096GB"]
- AMD MI210 GPU with 64 units
- AMD MI300X GPU with 16 units


## Training Parameters

Run the `train_qwen.py` script again without changing the batch size.

```bash
~/quickstart$ python tutorial/train_qwen.py
```

If the training proceeds normally, you should see the following logs:

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

Compared to the previous execution results when the number of GPUs was half, you can see that the learning is the same and the throughput has improved. 

- When using AMD MI250 GPU 16 → 32 : approximately 59,000 tokens/sec → 105,000 tokens/sec