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

Please contact your infrastructure provider and choose one of the following options before proceeding. 

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
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[warning] A newer version of Moreh AI Framework is available. You can update the software to the latest version by running "update-moreh".
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.1:24163)...
[info] [2/8] Connecting to resources on the server (192.168.110.2:24163)...
[info] [3/8] Connecting to resources on the server (192.168.110.4:24163)...
[info] [4/8] Connecting to resources on the server (192.168.110.37:24163)...
[info] [5/8] Connecting to resources on the server (192.168.110.39:24163)...
[info] [6/8] Connecting to resources on the server (192.168.110.40:24163)...
[info] [7/8] Connecting to resources on the server (192.168.110.72:24163)...
[info] [8/8] Connecting to resources on the server (192.168.110.73:24163)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 977770
[info] The number of candidates is 6.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 4578 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 2.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 254 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] num_stages : 4, num_micro_batches : 2, batch_per_device : 1, No TP, recomputation : false, distribute_param : true
[info] train: true
| INFO     | main:train:82 - [Step 0/3320] Loss: 0.9375 Throughput: 1765.73 tokens/sec
| INFO     | main:train:82 - [Step 10/3320] Loss: 0.6875 Throughput: 13705.69 tokens/sec
| INFO     | main:train:82 - [Step 20/3320] Loss: 0.66796875 Throughput: 13531.69 tokens/sec
| INFO     | main:train:82 - [Step 30/3320] Loss: 0.55078125 Throughput: 13839.31 tokens/sec
...
```

Compared to the previous execution results when the number of GPUs was half, you can see that the learning is the same and the throughput has improved. 

- When using AMD MI250 GPU 16 → 32 : approximately 6,800 tokens/sec → 13,000 tokens/sec
