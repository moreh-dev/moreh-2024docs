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
[KT Hyperscale AI Computing (HAC) AI Accelerator Information](/Supported_Documents/KT_HAC_Models_Info.md)

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
[2024-05-13 18:04:23.681] [info] Got DBs from backend for auto config.
[2024-05-13 18:04:25.833] [info] Requesting resources for KT AI Accelerator from the server...
[2024-05-13 18:04:25.844] [info] Initializing the worker daemon for KT AI Accelerator
[2024-05-13 18:04:30.622] [info] [1/8] Connecting to resources on the server (192.168.110.4:24172)...
[2024-05-13 18:04:30.637] [info] [2/8] Connecting to resources on the server (192.168.110.5:24172)...
[2024-05-13 18:04:30.645] [info] [3/8] Connecting to resources on the server (192.168.110.10:24172)...
[2024-05-13 18:04:30.651] [info] [4/8] Connecting to resources on the server (192.168.110.42:24172)...
[2024-05-13 18:04:30.658] [info] [5/8] Connecting to resources on the server (192.168.110.43:24172)...
[2024-05-13 18:04:30.665] [info] [6/8] Connecting to resources on the server (192.168.110.44:24172)...
[2024-05-13 18:04:30.672] [info] [7/8] Connecting to resources on the server (192.168.110.83:24172)...
[2024-05-13 18:04:30.679] [info] [8/8] Connecting to resources on the server (192.168.110.84:24172)...
[2024-05-13 18:04:30.686] [info] Establishing links to the resources...
[2024-05-13 18:04:31.612] [info] KT AI Accelerator is ready to use.
[2024-05-13 18:04:31.612] [info] Moreh Version: 24.5.0
[2024-05-13 18:04:31.612] [info] Moreh Job ID: 976907
[2024-05-13 18:04:31.835] [warning] Various batch size detected : 512, 1
[2024-05-13 18:04:31.835] [info] The number of candidates is 6.
[2024-05-13 18:04:31.835] [info] Parallel Graph Compile start...
[2024-05-13 18:04:32.468] [info] Elapsed Time to compile all candidates = 633 [ms]
[2024-05-13 18:04:32.469] [info] Parallel Graph Compile finished.
[2024-05-13 18:04:32.469] [info] The number of possible candidates is 2.
[2024-05-13 18:04:32.469] [info] SelectBestGraphFromCandidates start...
[2024-05-13 18:04:32.707] [info] Elapsed Time to compute cost for survived candidates = 238 [ms]
[2024-05-13 18:04:32.708] [info] SelectBestGraphFromCandidates finished.
[2024-05-13 18:04:32.708] [info] Configuration for parallelism is selected.
[2024-05-13 18:04:32.708] [info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : false
[2024-05-13 18:04:32.708] [info] train: true
2024-05-13 18:08:43.443 | INFO     | **main**:main:135 - [Step 2/560] | Loss: 2.1875 | Duration: 1.89 | Throughput: 276803.38 tokens/sec
2024-05-13 18:09:01.651 | INFO     | **main**:main:135 - [Step 4/560] | Loss: 2.109375 | Duration: 1.40 | Throughput: 375362.04 tokens/sec
2024-05-13 18:09:19.639 | INFO     | **main**:main:135 - [Step 6/560] | Loss: 2.046875 | Duration: 1.16 | Throughput: 450234.51 tokens/sec
2024-05-13 18:09:37.844 | INFO     | **main**:main:135 - [Step 8/560] | Loss: 2.015625 | Duration: 1.35 | Throughput: 387487.33 tokens/sec
2024-05-13 18:09:55.952 | INFO     | **main**:main:135 - [Step 10/560] | Loss: 2.015625 | Duration: 1.33 | Throughput: 393661.22 tokens/sec
...
```

If the training proceeds normally, you will see similar logs to the previous run but with improved throughput due to the doubled number of GPUs.

- When using AMD MI250 GPU 16 → 32 : From approximately 200,000 tokens/sec to 390,000 tokens/sec.

