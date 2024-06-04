---
icon: terminal
tags:  [tutorial, llama2]
order: 40
---

# 3. Model Fine-tuning
Now, we will train the model through the following process. 

## Setting Accelerator Flavor
In MoAI Platform, physical GPUs are not directly exposed to users. Instead, virtual MoAI Accelerators are provided, which are available for use in PyTorch. By setting the accelerator's flavor, you can determine how much of the physical GPU will be utilized by PyTorch. Since the total training time and GPU usage cost vary depending on the selected accelerator flavor, users should make decisions based on their training scenarios. If needed, please refer to the [LLM Fine-tuning Parameter Guide](/Supported_Documents/LLM_param_guide.md) to select the accelerator Flavor that aligns with your training objectives.

!!!
Please refer to the document above or reach out to your infrastructure provider to inquire about the GPU types and quantities corresponding to each flavor.
!!!


Before continuing with the tutorial, we recommend reaching out to your infrastructure provider to inquire about the types and quantities of GPUs associated with each flavor. Once you have this information, you can choose one of the following flavors to proceed:

- AMD MI250 GPU with 16 units:
    - Select [!badge variant="secondary" text="4xlarge"] when using Moreh's trial container.
    - Select [!badge variant="secondary" text="4xLarge.2048GB"] when using KT Cloud's Hyperscale AI Computing.
- AMD MI210 GPU with 32 units.
- AMD MI300X GPU with 8 units.

Remember when we checked the MoAI Accelerator in the [**Llama2 13B Fine-tuning - Getting Started**](1_Prepare_Fine-tuning.md)? Now let's set up the accelerator needed for learning.

First, we'll use the **`moreh-smi`** command to check the currently used MoAI Accelerator.

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

You can enter the number to switch to a different flavor.

In this tutorial, we will use a 2048GB-sized MoAI Accelerator.

Therefore, after switching from the initially set [!badge variant="secondary" text="Large.256GB"] flavor to [!badge variant="secondary" text="4xLarge.2048GB"], we will use the **`moreh-smi`** command to confirm that the change has been successfully applied.

Enter 8 to use [!badge variant="secondary" text="4xLarge.2048GB"]


```bash
Selection (1-13, q, Q): 8
The MoAI Accelerator model is successfully switched to  "4xLarge.2048GB".

1. Small.64GB
2. Medium.128GB
3. Large.256GB
4. xLarge.512GB
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB  *
9. 6xLarge.3072GB
10. 8xLarge.4096GB
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
|  * 0     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

Now you can see that it has been successfully changed to [!badge variant="secondary" text="4xLarge.2048GB"].

## Training Execution

Execute the given **`train_llama2.py`** script.

```bash
$ cd ~/quickstart
~/quickstart$ python tutorial/train_llama2.py
```

If the training proceeds smoothly, you should see the following logs. By going through this logs, you can verify that the Advanced Parallelism feature, which determines the optimal parallelization settings, is functioning properly. It's worth noting that, apart from the single line of AP code we looked at earlier in the PyTorch script, there is no handling for using multiple GPUs simultaneously in other parts of the script.

```
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/4] Connecting to resources on the server (192.168.110.1:24169)...
[info] [2/4] Connecting to resources on the server (192.168.110.26:24169)...
[info] [3/4] Connecting to resources on the server (192.168.110.61:24169)...
[info] [4/4] Connecting to resources on the server (192.168.110.88:24169)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 977753
[info] The number of candidates is 54.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 80292 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 37.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 49599 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : true
[info] train: true
| INFO     | __main__:main:136 - [Step 2/1121] | Loss: 1.84375 | Duration: 2.37 | Throughput: 110643.89 tokens/sec
| INFO     | __main__:main:136 - [Step 4/1121] | Loss: 1.78125 | Duration: 1.66 | Throughput: 158266.52 tokens/sec
| INFO     | __main__:main:136 - [Step 6/1121] | Loss: 1.71875 | Duration: 1.70 | Throughput: 154598.70 tokens/sec
| INFO     | __main__:main:136 - [Step 8/1121] | Loss: 1.734375 | Duration: 1.82 | Throughput: 143925.00 tokens/sec
...

Training Done
Saving Model...
Model saved in ./llama2_summarization
```

You can verify that the training is proceeding smoothly by checking the training logs.

The throughput displayed during training indicates how many tokens are being trained per second through the PyTorch script.

- Throughput when using 16 AMD MI250 GPUs: Approximately 150,000 tokens/sec

Here are the approximate training times based on the type and number of GPUs.

- Training time when using 16 AMD MI250 GPUs: Approximately 4 hours

## Checking Accelerator Status During Training

During training, open another terminal and connect to the container. Then, execute the **`moreh-smi`** command to observe the MoAI Accelerator occupying memory and the training script running. Make sure to check this while the initialization process is completed and the training loss appears in the execution logs.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Flavor     |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  4xLarge.2048GB  |  1354751 MiB   |  2096640 MiB   |     100%      |
+-----------------------------------------------------------------------------------------------------+

Processes:
+--------------------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID    |             Process               |  Memory Usage  |
+======================================================================================+
|       0  |  977753  |  2200972  |  python tutorial/train_llama2.py  |  1354751 MiB   |
+--------------------------------------------------------------------------------------+
```
