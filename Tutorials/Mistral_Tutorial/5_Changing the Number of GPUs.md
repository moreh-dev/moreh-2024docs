---
icon: terminal
tags: [tutorial, mistral]
order: 40
---

# 5. Changing the Number of GPUs

Let's rerun the fine-tuning task with a different number of GPUs. MoAI Platform abstracts GPU resources into a single accelerator and automatically performs parallel processing. Therefore, there is no need to modify the PyTorch script even when changing the number of GPUs.

## Changing the Accelerator type

Switch the accelerator type using the **`moreh-switch-model`** tool. For instructions on changing the accelerator, please refer to the [**3. Model fine-tuning**](3_fine_tuning.md) 

```bash
$ moreh-switch-model
```

Please contact your infrastructure provider and choose one of the following options before proceeding.

- AMD MI250 GPU with 32 units
    - When using Moreh's trial container: select [!badge variant="secondary" text=8xlarge] 
    - When using KT Cloud's Hyperscale AI Computing: select [!badge variant="secondary" text=8xLarge.4096GB]
- AMD MI210 GPU with 64 units
- AMD MI300X GPU with 16 units



## Training Parameters

Run the `train_mistral.py` script again without changing the batch size.

```bash
~/moreh-quickstart$ python tutorial/train_mistral.py
```

If the training proceeds normally, you should see the following logs:

```bash
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.10:24174)...
[info] [2/8] Connecting to resources on the server (192.168.110.33:24174)...
[info] [3/8] Connecting to resources on the server (192.168.110.34:24174)...
[info] [4/8] Connecting to resources on the server (192.168.110.52:24174)...
[info] [5/8] Connecting to resources on the server (192.168.110.53:24174)...
[info] [6/8] Connecting to resources on the server (192.168.110.79:24174)...
[info] [7/8] Connecting to resources on the server (192.168.110.80:24174)...
[info] [8/8] Connecting to resources on the server (192.168.110.98:24174)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] The number of candidates is 22.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 28502 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 4.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 990 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] num_stages : 2, num_micro_batches : 8, batch_per_device : 1, No TP, recomputation : false, distribute_param : true
[info] train: true

| INFO     | __main__:main:137 - [Step 1/144] | Loss: 1.1953125 | Duration: 41.70 | Throughput: 12572.33 tokens/sec
| INFO     | __main__:main:137 - [Step 2/144] | Loss: 0.85546875 | Duration: 4.68 | Throughput: 111965.54 tokens/sec
| INFO     | __main__:main:137 - [Step 3/144] | Loss: 0.796875 | Duration: 5.81 | Throughput: 90209.43 tokens/sec
| INFO     | __main__:main:137 - [Step 4/144] | Loss: 0.75390625 | Duration: 5.80 | Throughput: 90425.87 tokens/sec
| INFO     | __main__:main:137 - [Step 5/144] | Loss: 0.64453125 | Duration: 4.38 | Throughput: 119712.50 tokens/sec
...
```

Compared to the previous execution results when the number of GPUs was half, you can see that the learning is the same and the throughput has improved. 

- When using AMD MI250 GPU 16 → 32 : approximately 60,000 tokens/sec → 110,000 tokens/sec
