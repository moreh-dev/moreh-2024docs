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

Since the available GPU memory has doubled, let's increase the batch size from the previous 256 to 512 and run the code again.

```bash
~/quickstart$ python tutorial/train_qwen.py --batch-size 512
```

If the training proceeds normally, you should see the following logs:

```bash
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.15:24169)...
[info] [2/8] Connecting to resources on the server (192.168.110.17:24169)...
[info] [3/8] Connecting to resources on the server (192.168.110.18:24169)...
[info] [4/8] Connecting to resources on the server (192.168.110.42:24169)...
[info] [5/8] Connecting to resources on the server (192.168.110.44:24169)...
[info] [6/8] Connecting to resources on the server (192.168.110.45:24169)...
[info] [7/8] Connecting to resources on the server (192.168.110.79:24169)...
[info] [8/8] Connecting to resources on the server (192.168.110.80:24169)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 977793
[info] The number of candidates is 78.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 158950 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 66.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 85792 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : false
[info] train: true
| INFO     | __main__:main:132 - [Step 1/36] | Loss: 1.1484375 | Duration: 309.16 | Throughput: 1695.87 tokens/sec
| INFO     | __main__:main:132 - [Step 2/36] | Loss: 1.078125 | Duration: 1.31 | Throughput: 399581.94 tokens/sec
| INFO     | __main__:main:132 - [Step 3/36] | Loss: 1.0 | Duration: 1.35 | Throughput: 388552.67 tokens/sec
| INFO     | __main__:main:132 - [Step 4/36] | Loss: 0.8125 | Duration: 1.36 | Throughput: 386721.16 tokens/sec
| INFO     | __main__:main:132 - [Step 5/36] | Loss: 0.74609375 | Duration: 1.38 | Throughput: 380145.21 tokens/sec
...
```

Compared to the previous execution results when the number of GPUs was half, you can see that the learning is the same and the throughput has improved. 

- When using AMD MI250 GPU 16 → 32 : approximately 190,000 tokens/sec → 380,000 tokens/sec
