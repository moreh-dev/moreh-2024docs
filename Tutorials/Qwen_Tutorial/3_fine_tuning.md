---
icon: terminal
tags: [tutorial, qwen]
order: 40
---

# 3. Model Fine-tuning

Now, we will train the model through the following process. 

## Setting Accelerator Flavor

In MoAI Platform, physical GPUs are not directly exposed to users. Instead, virtual MoAI Accelerators are provided, which are available for use in PyTorch. By setting the accelerator's flavor, you can determine how much of the physical GPU will be utilized by PyTorch. Since the total training time and GPU usage cost vary depending on the selected accelerator flavor, users should make decisions based on their training scenarios. Refer to the following document to select the accelerator Flavor that aligns with your training objectives.


- [LLM Fine-tuning Parameter Guide](/Supported_Documents/LLM_param_guide.md)

!!!
Please refer to the document above or reach out to your infrastructure provider to inquire about the GPU types and quantities corresponding to each flavor.
!!!


You can choose one of the following flavors to proceed:

- AMD MI250 GPU with 16 units:
    - Select [!badge variant="secondary" text=4xlarge] when using Moreh's trial container.
    - Select [!badge variant="secondary" text=4xLarge.2048GB] when using KT Cloud's Hyperscale AI Computing.
- AMD MI210 GPU with 32 units.
- AMD MI300X GPU with 8 units.

**Do you remember checking MoAI Accelerator in the [Mistral Fine-tuning (ENG)](index.md) document? Now let's set up the accelerator needed for learning.**

First, we'll use the **`moreh-smi`** command to check the currently used MoAI Accelerator.

```bash
$ moreh-smi
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.2.0  Latest Version: 24.2.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  Small.64GB  |  -             |  -             |  -            |
+-------------------------------------------------------------------------------------------------+
```

The current MoAI Accelerator in use has a memory size of 64GB.

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

Therefore, after switching from the initially set [!badge variant="secondary" text=Small.64GB] flavor to [!badge variant="secondary" text=4xLarge.2048GB] , we will use the **`moreh-smi`** command to confirm that the change has been successfully applied.

Enter 8 to use [!badge variant="secondary" text=4xLarge.2048GB] 

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
|                                                    Current Version: 24.2.0  Latest Version: 24.2.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

Now you can see that it has been successfully changed to [!badge variant="secondary" text=4xLarge.2048GB].

## Training Execution

Execute the **`train_qwen.py`** script below.

```bash
$ cd ~/quickstart
~/quickstart$ python tutorial/train_qwen.py
```

If the training proceeds smoothly, you should see the following logs. By going through this logs, you can verify that the Advanced Parallelism feature, which determines the optimal parallelization settings, is functioning properly. It's worth noting that, apart from the single line of AP code we looked at earlier in the PyTorch script, there is no handling for using multiple GPUs simultaneously in other parts of the script.

```bash
...
[info] Got DBs from backend for auto config.
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/4] Connecting to resources on the server (192.168.110.5:24168)...
[info] [2/4] Connecting to resources on the server (192.168.110.23:24168)...
[info] [3/4] Connecting to resources on the server (192.168.110.45:24168)...
[info] [4/4] Connecting to resources on the server (192.168.110.73:24168)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] The number of candidates is 16.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 46137 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 4.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 1271 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] num_stages : 2, num_micro_batches : 16, batch_per_device : 1, No TP, recomputation : false, distribute_param : true
[info] train: true

| INFO     | __main__:main:150 - [Step 1/144] | Loss: 1.34375 | Duration: 57.79 | Throughput: 9072.25 tokens/sec
| INFO     | __main__:main:150 - [Step 2/144] | Loss: 0.83984375 | Duration: 7.42 | Throughput: 70685.21 tokens/sec
| INFO     | __main__:main:150 - [Step 3/144] | Loss: 0.9921875 | Duration: 10.37 | Throughput: 50536.92 tokens/sec
| INFO     | __main__:main:150 - [Step 4/144] | Loss: 0.98828125 | Duration: 9.84 | Throughput: 53281.45 tokens/sec
| INFO     | __main__:main:150 - [Step 5/144] | Loss: 0.75390625 | Duration: 10.41 | Throughput: 50347.46 tokens/sec
| INFO     | __main__:main:150 - [Step 6/144] | Loss: 0.69921875 | Duration: 10.60 | Throughput: 49452.14 tokens/sec
| INFO     | __main__:main:150 - [Step 7/144] | Loss: 0.53515625 | Duration: 10.65 | Throughput: 49214.62 tokens/sec
| INFO     | __main__:main:150 - [Step 8/144] | Loss: 0.609375 | Duration: 7.67 | Throughput: 68339.57 tokens/sec
| INFO     | __main__:main:150 - [Step 9/144] | Loss: 0.482421875 | Duration: 10.43 | Throughput: 50256.04 tokens/sec
...

Training Done
Saving Model...
Model saved in ./qwen_code_generation
```

You can confirm that the training is progressing smoothly by observing the loss values appearing as follows.

![](loss.png)

The throughput displayed during training indicates how many tokens per second are being processed through the PyTorch script.

- When using 16 AMD MI250 GPUs: approximately 59,000 tokens/sec

Approximate training time based on GPU type and quantity is as follows:

- When using 16 AMD MI250 GPUs: approximately 40 minutes

## Checking Accelerator Status During Training

During training, open another terminal and connect to the container. You can execute the `moreh-smi` command to observe the MoAI Accelerator occupying memory while the training script is running. Please check the memory occupancy of MoAI accelerator when the training loss appears in the execution log after the initialization process.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.2.0  Latest Version: 24.2.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |   MoAI Accelerator  |  4xLarge.2048GB  |  853677 MiB    |  2096640 MiB   |  100 %        |
+-----------------------------------------------------------------------------------------------------+

Processes:
+------------------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID    |             Process             |  Memory Usage  |
+====================================================================================+
|       0  |  975583  |  4090049  |  python tutorial/train_qwen.py  |  853677 MiB    |
+------------------------------------------------------------------------------------+
```

