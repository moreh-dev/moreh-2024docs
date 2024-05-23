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

Please contact your infrastructure provider and choose one of the following options before proceeding. [**KT Hyperscale AI Computing(HAC) AI Accelerator Information**](/Supported_Documents/KT_HAC_Models_Info.md)

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

Since the available GPU memory has doubled, let's increase the batch size to 2048 and run the training.

f the training proceeds normally, you should see the following log:



```bash
$ python tutorial/train_baichuan2_13b.py
[2024-04-26 18:58:21.192] [info] Requesting resources for KT AI Accelerator from the server...
[2024-04-26 18:58:21.203] [warning] A newer version of Moreh AI Framework is available. You can update the software to the latest version by running "update-moreh".
[2024-04-26 18:58:21.203] [info] Initializing the worker daemon for KT AI Accelerator
[2024-04-26 18:58:26.191] [info] [1/8] Connecting to resources on the server (192.168.110.32:24174)...
[2024-04-26 18:58:26.206] [info] [2/8] Connecting to resources on the server (192.168.110.33:24174)...
[2024-04-26 18:58:26.215] [info] [3/8] Connecting to resources on the server (192.168.110.35:24174)...
[2024-04-26 18:58:26.221] [info] [4/8] Connecting to resources on the server (192.168.110.67:24174)...
[2024-04-26 18:58:26.228] [info] [5/8] Connecting to resources on the server (192.168.110.73:24174)...
[2024-04-26 18:58:26.233] [info] [6/8] Connecting to resources on the server (192.168.110.75:24174)...
[2024-04-26 18:58:26.239] [info] [7/8] Connecting to resources on the server (192.168.110.97:24174)...
[2024-04-26 18:58:26.245] [info] [8/8] Connecting to resources on the server (192.168.110.98:24174)...
[2024-04-26 18:58:26.251] [info] Establishing links to the resources...
[2024-04-26 18:58:27.115] [info] KT AI Accelerator is ready to use.
[2024-04-26 18:58:27.381] [info] The number of candidates is 65.
[2024-04-26 18:58:27.381] [info] Parallel Graph Compile start...
[2024-04-26 18:58:38.857] [info] Elapsed Time to compile all candidates = 11476 [ms]
[2024-04-26 18:58:38.857] [info] Parallel Graph Compile finished.
[2024-04-26 18:58:38.857] [info] The number of possible candidates is 6.
[2024-04-26 18:58:38.857] [info] SelectBestGraphFromCandidates start...
[2024-04-26 18:58:40.330] [info] Elapsed Time to compute cost for survived candidates = 1472 [ms]
[2024-04-26 18:58:40.330] [info] SelectBestGraphFromCandidates finished.
[2024-04-26 18:58:40.330] [info] Configuration for parallelism is selected.
[2024-04-26 18:58:40.330] [info] No PP, No TP, recomputation : 1, distribute_param : true, distribute_low_prec_param : true
[2024-04-26 18:58:40.330] [info] train: true

2024-04-26 19:05:47.509 | INFO     | __main__:main:143 - [Step 1/52] Throughput : 1167.211504009616tokens/sec
2024-04-26 19:05:49.065 | INFO     | __main__:main:143 - [Step 2/52] Throughput : 358524.96263602894tokens/sec
2024-04-26 19:05:50.598 | INFO     | __main__:main:143 - [Step 3/52] Throughput : 380980.5659610025tokens/sec
2024-04-26 19:05:52.088 | INFO     | __main__:main:143 - [Step 4/52] Throughput : 382460.244826232tokens/sec
2024-04-26 19:05:53.628 | INFO     | __main__:main:143 - [Step 5/52] Throughput : 377403.73612910055tokens/sec
2024-04-26 19:05:55.191 | INFO     | __main__:main:143 - [Step 6/52] Throughput : 382224.183245965tokens/sec
2024-04-26 19:05:56.813 | INFO     | __main__:main:143 - [Step 7/52] Throughput : 380014.4669324378tokens/sec
```

Upon comparison with the results obtained when the number of GPUs was halved, you'll notice that the training progresses similarly, with an improvement in throughput.

- When using AMD MI250 GPU 16 → 32 : approximately 198,000 tokens/sec → 370,000 tokens/sec