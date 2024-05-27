---
icon: terminal
tags: [tutorial, llama2]
order: 40
---

# 5. Changing the Number of GPUs

Let's rerun the fine-tuning task with a different number of GPUs. MoAI Platform abstracts GPU resources into a single accelerator and automatically performs parallel processing. Therefore, there is no need to modify the PyTorch script even when changing the number of GPUs.


## Changing the Accelerator type

Switch the accelerator type using the **`moreh-switch-model`** tool. For instructions on changing the accelerator, please refer again to the [3. Model fine-tuning](3_fine_tuning.md).

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
Again, run the **`train_llama3.py`** script.

```bash
~/quickstart$ python tutorial/train_llama3.py --batch-size 512
```

Since the available GPU memory has doubled, let's increase the batch size from the previous 256 to 512 and run the code again.



```bash
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.4:24172)...
[info] [2/8] Connecting to resources on the server (192.168.110.5:24172)...
[info] [3/8] Connecting to resources on the server (192.168.110.10:24172)...
[info] [4/8] Connecting to resources on the server (192.168.110.42:24172)...
[info] [5/8] Connecting to resources on the server (192.168.110.43:24172)...
[info] [6/8] Connecting to resources on the server (192.168.110.44:24172)...
[info] [7/8] Connecting to resources on the server (192.168.110.83:24172)...
[info] [8/8] Connecting to resources on the server (192.168.110.84:24172)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 976907
[warning] Various batch size detected : 512, 1
[info] The number of candidates is 6.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 633 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 2.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 238 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : false
[info] train: true
| INFO     | main:main:135 - [Step 2/560] | Loss: 2.1875 | Duration: 1.89 | Throughput: 276803.38 tokens/sec
| INFO     | main:main:135 - [Step 4/560] | Loss: 2.109375 | Duration: 1.40 | Throughput: 375362.04 tokens/sec
| INFO     | main:main:135 - [Step 6/560] | Loss: 2.046875 | Duration: 1.16 | Throughput: 450234.51 tokens/sec
| INFO     | main:main:135 - [Step 8/560] | Loss: 2.015625 | Duration: 1.35 | Throughput: 387487.33 tokens/sec
| INFO     | main:main:135 - [Step 10/560] | Loss: 2.015625 | Duration: 1.33 | Throughput: 393661.22 tokens/sec
...
```

If the training proceeds normally, you will see similar logs to the previous run but with improved throughput due to the doubled number of GPUs.

- When using AMD MI250 GPU 16 → 32 : From approximately 200,000 tokens/sec to 390,000 tokens/sec.

