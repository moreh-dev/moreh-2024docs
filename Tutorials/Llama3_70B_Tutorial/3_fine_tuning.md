---
icon: terminal
tags:  [tutorial, llama3_70b]
order: 40
---

# 3. Model Fine-tuning
Now, we will actually execute the fine-tuning process.

## Setting Accelerator Flavor

In MoAI Platform, physical GPUs are not directly exposed to users. Instead, virtual MoAI Accelerators are provided, which are available for use in PyTorch. By setting the accelerator's flavor, you can determine how much of the physical GPU will be utilized by PyTorch. Since the total training time and GPU usage cost vary depending on the selected accelerator flavor, users should make decisions based on their training scenarios. If needed, please refer to the [LLM Fine-tuning Parameter Guide](/Supported_Documents/LLM_param_guide.md) to select the accelerator Flavor that aligns with your training objectives.

!!!
Please refer to the document above or reach out to your infrastructure provider  for the types and numbers of GPUs corresponding to each flavor.
!!!

Select one of the following flavors to continue:
- Using 32 AMD MI250 GPUs
    - Select [!badge variant="secondary" text="8xlarge"] when using Moreh's trial container.
    - Select [!badge variant="secondary" text="8xLarge.4096GB"] when using KT Cloud's Hyperscale AI Computing.
- Using 64 AMD MI210 GPUs
- Using 16 AMD MI300X GPUs

Remember when we checked the MoAI Accelerator in the [**Llama3 70B Fine-tuning - Getting Started**](index.md)? Now let's set up the accelerator needed for learning.

First, use the **`moreh-smi`** command to check the current MoAI Accelerator in use.

```bash
$ moreh-smi
+---------------------------------------------------------------------------------------------------+
|                                                  Current Version: 24.5.0  Latest Version: 24.5.0  |
+---------------------------------------------------------------------------------------------------+
|  Device  |        Name         |      Model     |  Memory Usage  |  Total Memory  |  Utilization  |
+===================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+---------------------------------------------------------------------------------------------------+
```

The current MoAI Accelerator in use has a memory size of 512GB.

You can utilize the `moreh-switch-model` command to review the available accelerator flavors on the current system. For seamless model training, consider using the `moreh-switch-model`command to switch to a MoAI Accelerator with larger memory capacity.

```bash
$ moreh-switch-model
Current MoAI Accelerator: xLarge.512GB

1. Small.64GB
2. Medium.128GB
3. Large.256GB
4. xLarge.512GB  *
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB
9. 6xLarge.3072GB
10. 8xLarge.4096GB
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB
```

You can enter a number here to switch to a different flavor.

For this tutorial, we will use the 4096GB MoAI Accelerator.

Therefore, we will switch the initially set [!badge variant="secondary" text="xLarge.512GB"] flavor to [!badge variant="secondary" text="8xLarge.4096GB"] and then use the **`moreh-smi`** command to verify that the change has been applied correctly.

Enter **`10`** to use [!badge variant="secondary" text="8xLarge.4096GB"].


```bash
Selection (1-13, q, Q): 10
The MoAI Accelerator model is successfully switched to  "8xLarge.4096GB".

1. Small.64GB
2. Medium.128GB
3. Large.256GB
4. xLarge.512GB
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB
9. 6xLarge.3072GB
10. 8xLarge.4096GB  *
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB

Selection (1-13, q, Q): q 
```

Enter **`q`** to complete the change.

To confirm that the changes have been successfully applied, use the **`moreh-smi`** command again to check the currently used MoAI Accelerator.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |   MoAI Accelerator  |  8xLarge.4096GB  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

You can see that it has been successfully switched to [!badge variant="secondary" text="8xLarge.4096GB"].

## Training Execution
Run the provided **`train_llama3_70b.py`** script.

```bash
$ cd ~/quickstart
~/quickstart$ python tutorial/train_llama3_70b.py
```

If the training is running correctly, you will see logs similar to the following. These logs indicate that the Advanced Parallelism feature, which finds the optimal parallelization settings, is functioning correctly. Note that in the PyTorch script we reviewed earlier, no additional handling for using multiple GPUs simultaneously was necessary apart from the single AP code line.

```
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/8] Connecting to resources on the server (192.168.110.17:24169)...
[info] [2/8] Connecting to resources on the server (192.168.110.18:24169)...
[info] [3/8] Connecting to resources on the server (192.168.110.21:24169)...
[info] [4/8] Connecting to resources on the server (192.168.110.52:24169)...
[info] [5/8] Connecting to resources on the server (192.168.110.53:24169)...
[info] [6/8] Connecting to resources on the server (192.168.110.55:24169)...
[info] [7/8] Connecting to resources on the server (192.168.110.85:24169)...
[info] [8/8] Connecting to resources on the server (192.168.110.87:24169)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 977399
[info] The number of candidates is 78.
info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 383967 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 13.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 77913 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] num_stages : 8, num_micro_batches : 64, batch_per_device : 1, No TP, recomputation : full(2), distribute_param : true, distribute_low_prec_param : false
[info] train: true
| INFO     | __main__:main:154 - [Step 10/560] | Loss: 1.8515625 | Duration: 129.06 | Throughput: 4062.49 tokens/sec
| INFO     | __main__:main:154 - [Step 20/560] | Loss: 1.7890625 | Duration: 129.51 | Throughput: 4048.18 tokens/sec
| INFO     | __main__:main:154 - [Step 30/560] | Loss: 1.5716357 | Duration: 129.79 | Throughput: 4096.08 tokens/sec
| INFO     | __main__:main:154 - [Step 40/560] | Loss: 1.6547084 | Duration: 128.72 | Throughput: 4124.73 tokens/sec
...
Training Done
Saving Model...
Model saved in ./llama3_70b_summarization
```

From the training logs, you can confirm that the training is progressing smoothly.

The throughput displayed during training indicates the number of tokens being trained per second by the PyTorch script.

- Using 32 AMD MI250 GPUs (64 devices): approximately 4062 tokens/sec

Estimated training time based on GPU type and count is as follows:

- Using 32 AMD MI250 GPUs (64 devices): approximately 24 hours

## Checking Accelerator Status During Training

During training, you can open another terminal and connect to the container. Then, run the **`moreh-smi`** command to see the MoAI Accelerator’s memory being utilized by the training script, as shown below.

```bash
$ moreh-smi
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Flavor     |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  8xLarge.2048GB  |  3516969 MiB   |  4193280 MiB   |    100%       |
+-----------------------------------------------------------------------------------------------------+
```
