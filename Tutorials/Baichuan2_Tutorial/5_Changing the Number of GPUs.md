---
icon: terminal
tags: [tutorial, baichuan]
order: 40
---

# 5. Changing the Number of GPUs

Let's rerun the fine-tuning task with a different number of GPUs. MoAI Platform abstracts GPU resources into a single accelerator and automatically performs parallel processing. Therefore, there is no need to modify the PyTorch script even when changing the number of GPUs.

## Changing the Accelerator type

Switch the accelerator type using the `moreh-switch-model` tool. For instructions on changing the accelerator, please refer to the [**3. Model fine-tuning**](3_finetuning.md)

```
$ moreh-switch-model
```

Please contact your infrastructure provider and choose one of the following options before proceeding.

- AMD MI250 GPU with 32 units
    - When using Moreh's trial container: select [!badge variant="secondary" text="8xlarge"]
    - When using KT Cloud's Hyperscale AI Computing: select [!badge variant="secondary" text="8xLarge.4096GB"] 
- AMD MI210 GPU with 64 units
- AMD MI300X GPU with 16 units

## Training Parameters

Run the `train_baichuan2_13b.py` script again.

```bash
~/moreh-quickstart$ python tutorial/train_baichuan2_13b.py --batch-size 512
```

Since the available GPU memory has doubled, let's increase the batch size to 512 and run the training.

f the training proceeds normally, you should see the following log:



```bash
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.32:24174)...
[info] [2/8] Connecting to resources on the server (192.168.110.33:24174)...
[info] [3/8] Connecting to resources on the server (192.168.110.35:24174)...
[info] [4/8] Connecting to resources on the server (192.168.110.67:24174)...
[info] [5/8] Connecting to resources on the server (192.168.110.73:24174)...
[info] [6/8] Connecting to resources on the server (192.168.110.75:24174)...
[info] [7/8] Connecting to resources on the server (192.168.110.97:24174)...
[info] [8/8] Connecting to resources on the server (192.168.110.98:24174)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] The number of candidates is 65.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 11476 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 6.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 1472 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] No PP, No TP, recomputation : 1, distribute_param : true, distribute_low_prec_param : true
[info] train: true

| INFO     | __main__:main:143 - [Step 1/52] Throughput : 1167.211504009616tokens/sec
| INFO     | __main__:main:143 - [Step 2/52] Throughput : 358524.96263602894tokens/sec
| INFO     | __main__:main:143 - [Step 3/52] Throughput : 380980.5659610025tokens/sec
| INFO     | __main__:main:143 - [Step 4/52] Throughput : 382460.244826232tokens/sec
| INFO     | __main__:main:143 - [Step 5/52] Throughput : 377403.73612910055tokens/sec
| INFO     | __main__:main:143 - [Step 6/52] Throughput : 382224.183245965tokens/sec
| INFO     | __main__:main:143 - [Step 7/52] Throughput : 380014.4669324378tokens/sec
...
```

Upon comparison with the results obtained when the number of GPUs was halved, you'll notice that the training progresses similarly, with an improvement in throughput.

- When using AMD MI250 GPU 16 → 32 : approximately 198,000 tokens/sec → 370,000 tokens/sec
