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

Since the GPU memory has doubled, let's increase the batch size from the previous 256 to 512 and run the code again.

If the training proceeds smoothly, you'll see logs similar to the following:

```bash
...
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.1:24166)...
[info] [2/8] Connecting to resources on the server (192.168.110.24:24166)...
[info] [3/8] Connecting to resources on the server (192.168.110.25:24166)...
[info] [4/8] Connecting to resources on the server (192.168.110.26:24166)...
[info] [5/8] Connecting to resources on the server (192.168.110.51:24166)...
[info] [6/8] Connecting to resources on the server (192.168.110.79:24166)...
[info] [7/8] Connecting to resources on the server (192.168.110.80:24166)...
[info] [8/8] Connecting to resources on the server (192.168.110.99:24166)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] The number of candidates is 24.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 57920 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 3.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 2120 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] num_stages : 2, num_micro_batches : 16, batch_per_device : 1, No TP, recomputation : true, distribute_param : true
[info] train: true

| INFO     | __main__:main:136 - [Step 2/560] | Loss: 1.6953125 | Duration: 15.12 | Throughput: 69337.98 tokens/sec
| INFO     | __main__:main:136 - [Step 4/560] | Loss: 1.6875 | Duration: 15.93 | Throughput: 65842.99 tokens/sec
| INFO     | __main__:main:136 - [Step 6/560] | Loss: 1.703125 | Duration: 15.73 | Throughput: 66656.28 tokens/sec
| INFO     | __main__:main:136 - [Step 8/560] | Loss: 1.6328125 | Duration: 15.36 | Throughput: 68263.53 tokens/sec
| INFO     | __main__:main:136 - [Step 10/560] | Loss: 1.5859375 | Duration: 12.78 | Throughput: 82040.81 tokens/sec
| INFO     | __main__:main:136 - [Step 12/560] | Loss: 1.59375 | Duration: 13.00 | Throughput: 80657.85 tokens/sec
| INFO     | __main__:main:136 - [Step 14/560] | Loss: 1.6328125 | Duration: 12.65 | Throughput: 82906.48 tokens/sec
| INFO     | __main__:main:136 - [Step 16/560] | Loss: 1.6796875 | Duration: 14.94 | Throughput: 70195.58 tokens/sec
| INFO     | __main__:main:136 - [Step 18/560] | Loss: 1.6875 | Duration: 13.01 | Throughput: 80607.34 tokens/sec
...
```

Compared to the previous results obtained when the GPU count was halved, you'll notice that the training is progressing similarly, but with an improved throughput.

- When using AMD MI250 GPU 16 → 32 : approximately 35,000 tokens/sec → 74,000 tokens/sec
