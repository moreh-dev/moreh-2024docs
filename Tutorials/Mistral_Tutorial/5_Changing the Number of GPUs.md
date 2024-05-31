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

Since the available GPU memory has doubled, let's increase the batch size from the previous 256 to 512 and run the code again.

```bash
~/moreh-quickstart$ python tutorial/train_mistral.py --batch-size 512
```

If the training proceeds normally, you should see the following logs:

```bash
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.13:24166)...
[info] [2/8] Connecting to resources on the server (192.168.110.17:24166)...
[info] [3/8] Connecting to resources on the server (192.168.110.18:24166)...
[info] [4/8] Connecting to resources on the server (192.168.110.42:24166)...
[info] [5/8] Connecting to resources on the server (192.168.110.44:24166)...
[info] [6/8] Connecting to resources on the server (192.168.110.45:24166)...
[info] [7/8] Connecting to resources on the server (192.168.110.79:24166)...
[info] [8/8] Connecting to resources on the server (192.168.110.80:24166)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 977805
[info] The number of candidates is 78.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 353868 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 66.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 167657 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : true
[info] train: true
| INFO     | __main__:main:131 - [Step 1/18] | Loss: 1.171875 | Duration: 635.97 | Throughput: 1648.78 tokens/sec
| INFO     | __main__:main:131 - [Step 2/18] | Loss: 0.88671875 | Duration: 1.34 | Throughput: 781064.78 tokens/sec
| INFO     | __main__:main:131 - [Step 3/18] | Loss: 0.72265625 | Duration: 1.35 | Throughput: 778463.12 tokens/sec
| INFO     | __main__:main:131 - [Step 4/18] | Loss: 0.625 | Duration: 1.34 | Throughput: 785383.61 tokens/sec
| INFO     | __main__:main:131 - [Step 5/18] | Loss: 0.5859375 | Duration: 1.30 | Throughput: 806700.33 tokens/sec
...
```

Compared to the previous execution results when the number of GPUs was half, you can see that the learning is the same and the throughput has improved. 

- When using AMD MI250 GPU 16 → 32 : approximately 390,000 tokens/sec → 800,000 tokens/sec
