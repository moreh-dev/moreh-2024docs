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
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.10:24155)...
[info] [2/8] Connecting to resources on the server (192.168.110.12:24155)...
[info] [3/8] Connecting to resources on the server (192.168.110.26:24155)...
[info] [4/8] Connecting to resources on the server (192.168.110.32:24155)...
[info] [5/8] Connecting to resources on the server (192.168.110.51:24155)...
[info] [6/8] Connecting to resources on the server (192.168.110.78:24155)...
[info] [7/8] Connecting to resources on the server (192.168.110.96:24155)...
[info] [8/8] Connecting to resources on the server (192.168.110.97:24155)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] The number of candidates is 22.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 28411 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 4.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 661 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] num_stages : 2, num_micro_batches : 8, batch_per_device : 1, No TP, recomputation : false, distribute_param : true
[info] train: true

| INFO     | __main__:main:154 - [Step 1/144] | Loss: 1.03125 | Duration: 39.51 | Throughput: 13270.26 tokens/sec
| INFO     | __main__:main:154 - [Step 2/144] | Loss: 1.0859375 | Duration: 4.37 | Throughput: 119959.56 tokens/sec
| INFO     | __main__:main:154 - [Step 3/144] | Loss: 0.8984375 | Duration: 4.44 | Throughput: 118024.31 tokens/sec
| INFO     | __main__:main:154 - [Step 4/144] | Loss: 0.85546875 | Duration: 5.30 | Throughput: 99006.09 tokens/sec
| INFO     | __main__:main:154 - [Step 5/144] | Loss: 0.890625 | Duration: 5.72 | Throughput: 91618.65 tokens/sec
...
| INFO     | __main__:main:154 - [Step 141/144] | Loss: 0.46875 | Duration: 5.28 | Throughput: 99212.01 tokens/sec
| INFO     | __main__:main:154 - [Step 142/144] | Loss: 0.45703125 | Duration: 4.35 | Throughput: 120536.34 tokens/sec
| INFO     | __main__:main:154 - [Step 143/144] | Loss: 0.451171875 | Duration: 4.35 | Throughput: 120554.52 tokens/sec
| INFO     | __main__:main:154 - [Step 144/144] | Loss: 0.443359375 | Duration: 4.41 | Throughput: 118957.15 tokens/sec
...
```

Compared to the previous execution results when the number of GPUs was half, you can see that the learning is the same and the throughput has improved. 

- When using AMD MI250 GPU 16 → 32 : approximately 59,000 tokens/sec → 105,000 tokens/sec
