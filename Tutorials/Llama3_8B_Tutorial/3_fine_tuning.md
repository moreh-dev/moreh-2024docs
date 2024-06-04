---
icon: terminal
tags:  [tutorial, llama3]
order: 40
---

# 3. Model Fine-tuning
Now, we will actually execute the fine-tuning process.

## Setting Accelerator Flavor

In MoAI Platform, physical GPUs are not directly exposed to users. Instead, virtual MoAI Accelerators are provided, which are available for use in PyTorch. By setting the accelerator's flavor, you can determine how much of the physical GPU will be utilized by PyTorch. Since the total training time and GPU usage cost vary depending on the selected accelerator flavor, users should make decisions based on their training scenarios. If needed, please refer to the [LLM Fine-tuning Parameter Guide](/Supported_Documents/LLM_param_guide.md) to select the accelerator Flavor that aligns with your training objectives.

!!!
Please refer to the document above or reach out to your infrastructure provider to inquire about the GPU types and quantities corresponding to each flavor.
!!!

- AMD MI250 GPU with 16 units:
    - Select [!badge variant="secondary" text="4xlarge"] when using Moreh's trial container.
    - Select [!badge variant="secondary" text="4xLarge.2048GB"] when using KT Cloud's Hyperscale AI Computing.
- AMD MI210 GPU with 32 units.
- AMD MI300X GPU with 8 units.

Remember when we checked the MoAI Accelerator in the [**Llama3 8B Fine-tuning - Getting Started**](index.md)? Now let's set up the accelerator needed for learning.

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

Execute the **`train_llama3.py`** script below.

```
$ cd ~/quickstart
~/quickstart$ python tutorial/train_llama3.py
```

If the training proceeds smoothly, you should see the following logs. By going through this logs, you can verify that the Advanced Parallelism feature, which determines the optimal parallelization settings, is functioning properly. It's worth noting that, apart from the single line of AP code we looked at earlier in the PyTorch script, there is no handling for using multiple GPUs simultaneously in other parts of the script.

```
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/4] Connecting to resources on the server (192.168.110.12:24160)...
[info] [2/4] Connecting to resources on the server (192.168.110.37:24160)...
[info] [3/4] Connecting to resources on the server (192.168.110.66:24160)...
[info] [4/4] Connecting to resources on the server (192.168.110.90:24160)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] Moreh Version: 24.5.0
[info] Moreh Job ID: 977876
[info] The number of candidates is 54.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 69128 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 44.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 32200 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : false
[info] train: true

| INFO     | __main__:main:161 - [Step 2/1121] | Loss: 2.015625 | Duration: 2.32 | Throughput: 113155.17 tokens/sec
| INFO     | __main__:main:161 - [Step 4/1121] | Loss: 1.921875 | Duration: 1.30 | Throughput: 201960.37 tokens/sec
| INFO     | __main__:main:161 - [Step 6/1121] | Loss: 1.921875 | Duration: 1.27 | Throughput: 206829.49 tokens/sec
| INFO     | __main__:main:161 - [Step 8/1121] | Loss: 1.9609375 | Duration: 1.30 | Throughput: 202362.30 tokens/sec
...

Training Done
Saving Model...
Model saved in ./llama3_summarization
```

Upon checking the training logs, you can confirm that the training is progressing smoothly.

The throughput displayed during training indicates how many tokens per second the script is training through this PyTorch script.

- When using AMD MI250 GPU with 16 GPUs: Approximately 200,000 tokens/sec

The approximate training time depending on the GPU type and quantity is as follows:

- When using AMD MI250 GPU with 16 GPUs: Approximately 160 minutes

## Checking Accelerator Status During Training

During training, open another terminal and connect to the container. Then, execute the **`moreh-smi`** command to observe the MoAI Accelerator occupying memory and the training script running. Make sure to check this while the initialization process is completed and the training loss appears in the execution logs.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Flavor     |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  4xLarge.2048GB  |  1472925 MiB   |  2096640 MiB   |    100%       |
+-----------------------------------------------------------------------------------------------------+
```
