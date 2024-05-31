---
icon: terminal
tags: [tutorial, llama2]
order: 40
---

# 5. Changing the Number of GPUs

Let's rerun the fine-tuning task with a different number of GPUs. MoAI Platform abstracts GPU resources into a single accelerator and automatically performs parallel processing. Therefore, there is no need to modify the PyTorch script even when changing the number of GPUs.


## Changing Accelerator type

Switch the accelerator type using the **`moreh-switch-model`** tool. For instructions on changing the accelerator, please refer again to the [3. Model fine-tuning](3_fine_tuning.md).

```bash
$ moreh-switch-model
```

- AMD MI250 GPU with 32 units
    - When using Moreh's trial container: select [!badge variant="secondary" text="8xlarge"]
    - When using KT Cloud's Hyperscale AI Computing: select [!badge variant="secondary" text="8xLarge.4096GB"]
- AMD MI210 GPU with 64 units
- AMD MI300X GPU with 16 units


## Training Parameters

Run the `train_llama2.py` script again.

```bash
~/quickstart$ python tutorial/train_llama2.py --batch-size 512
```

Since the GPU memory has doubled, let's increase the batch size from the previous `256` to `512` and run the code again.

If the training proceeds smoothly, you'll see logs similar to the following:

```bash
...
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.12:24162)...
[info] [2/8] Connecting to resources on the server (192.168.110.13:24162)...
[info] [3/8] Connecting to resources on the server (192.168.110.14:24162)...
[info] [4/8] Connecting to resources on the server (192.168.110.37:24162)...
[info] [5/8] Connecting to resources on the server (192.168.110.39:24162)...
[info] [6/8] Connecting to resources on the server (192.168.110.73:24162)...
[info] [7/8] Connecting to resources on the server (192.168.110.75:24162)...
[info] [8/8] Connecting to resources on the server (192.168.110.99:24162)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 977792
[info] The number of candidates is 78.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 190703 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 58.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 110073 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : true
[info] train: true

| INFO     | __main__:main:136 - [Step 2/560] | Loss: 1.8515625 | Duration: 3.62 | Throughput: 144909.28 tokens/sec
| INFO     | __main__:main:136 - [Step 4/560] | Loss: 1.765625 | Duration: 1.59 | Throughput: 328842.71 tokens/sec
| INFO     | __main__:main:136 - [Step 6/560] | Loss: 1.734375 | Duration: 1.51 | Throughput: 346245.79 tokens/sec
| INFO     | __main__:main:136 - [Step 8/560] | Loss: 1.703125 | Duration: 1.70 | Throughput: 307668.92 tokens/sec
...
```

Compared to the previous results obtained when the GPU count was halved, you'll notice that the training is progressing similarly, but with an improved throughput.

- When using AMD MI250 GPU 16 → 32 : approximately 150,000 tokens/sec → 315,000 tokens/sec
