---
icon: terminal
tags:  [tutorial, llama2]
order: 40
---

# 3. Model fine-tuning
Now, we will actually execute the fine-tuning process.

## Setting Accelerator Flavor

In MoAI Platform, physical GPUs are not directly exposed to users. Instead, virtual MoAI Accelerators are provided, which are available for use in PyTorch. By setting the accelerator's flavor, you can determine how much of the physical GPU will be utilized by PyTorch. Since the total training time and GPU usage cost vary depending on the selected accelerator flavor, users should make decisions based on their training scenarios. Refer to the following document to select the accelerator Flavor that aligns with your training objectives.

- **[KT Hyperscale AI Computing (HAC) AI Accelerator Information](/Supported_Documents/KT_HAC_Models_Info.md)**  
- **[LLM Fine-tuning Parameter Guide](/Supported_Documents/LLM_param_guide.md)**

!!!
Please refer to the document above or reach out to your infrastructure provider to inquire about the GPU types and quantities corresponding to each flavor.
!!!

- AMD MI250 GPU with 16 units:
    - Select [!badge variant="secondary" text="4xlarge"] when using Moreh's trial container.
    - Select [!badge variant="secondary" text="4xLarge.2048GB"] when using KT Cloud's Hyperscale AI Computing.
- AMD MI210 GPU with 32 units.
- AMD MI300X GPU with 8 units.

Remember when we checked the MoAI Accelerator in the [1. Llama3 8B Fine-tuning](index.md)? Now let's set up the accelerator needed for learning.

First, we'll use the **`moreh-smi`** command to check the currently used MoAI Accelerator.

```bash
$ moreh-smi
11:40:36 April 16, 2024
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.2.0  Latest Version: 24.2.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+-------------------------------------------------------------------------------------------------+
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
23:56:17 April 18, 2024
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.2.0  Latest Version: 24.2.0  |
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

```bash
[2024-05-13 17:52:41.897] [info] Got DBs from backend for auto config.
[2024-05-13 17:52:44.235] [info] Requesting resources for KT AI Accelerator from the server...
[2024-05-13 17:52:44.248] [info] Initializing the worker daemon for KT AI Accelerator
[2024-05-13 17:52:48.927] [info] [1/4] Connecting to resources on the server (192.168.110.39:24158)...
[2024-05-13 17:52:48.941] [info] [2/4] Connecting to resources on the server (192.168.110.40:24158)...
[2024-05-13 17:52:48.949] [info] [3/4] Connecting to resources on the server (192.168.110.80:24158)...
[2024-05-13 17:52:48.956] [info] [4/4] Connecting to resources on the server (192.168.110.81:24158)...
[2024-05-13 17:52:48.963] [info] Establishing links to the resources...
[2024-05-13 17:52:49.393] [info] KT AI Accelerator is ready to use.
[2024-05-13 17:52:49.393] [info] Moreh Version: 24.5.0
[2024-05-13 17:52:49.393] [info] Moreh Job ID: 976905
[2024-05-13 17:52:49.617] [warning] Various batch size detected : 256, 1
[2024-05-13 17:52:49.617] [info] The number of candidates is 6.
[2024-05-13 17:52:49.617] [info] Parallel Graph Compile start...
[2024-05-13 17:52:50.270] [info] Elapsed Time to compile all candidates = 652 [ms]
[2024-05-13 17:52:50.270] [info] Parallel Graph Compile finished.
[2024-05-13 17:52:50.270] [info] The number of possible candidates is 2.
[2024-05-13 17:52:50.270] [info] SelectBestGraphFromCandidates start...
[2024-05-13 17:52:50.450] [info] Elapsed Time to compute cost for survived candidates = 179 [ms]
[2024-05-13 17:52:50.450] [info] SelectBestGraphFromCandidates finished.
[2024-05-13 17:52:50.450] [info] Configuration for parallelism is selected.
[2024-05-13 17:52:50.450] [info] No PP, No TP, recomputation : default(1), distribute_param : true, distribute_low_prec_param : false
[2024-05-13 17:52:50.450] [info] train: true
2024-05-13 17:57:05.878 | INFO     | __main__:main:135 - [Step 2/1121] | Loss: 2.203125 | Duration: 4.65 | Throughput: 56416.87 tokens/sec
2024-05-13 17:57:23.075 | INFO     | __main__:main:135 - [Step 4/1121] | Loss: 2.046875 | Duration: 1.18 | Throughput: 221435.66 tokens/sec
2024-05-13 17:57:40.310 | INFO     | __main__:main:135 - [Step 6/1121] | Loss: 2.015625 | Duration: 1.20 | Throughput: 218975.01 tokens/sec
2024-05-13 17:57:57.427 | INFO     | __main__:main:135 - [Step 8/1121] | Loss: 2.015625 | Duration: 1.14 | Throughput: 229897.61 tokens/sec
2024-05-13 17:58:14.628 | INFO     | __main__:main:135 - [Step 10/1121] | Loss: 2.015625 | Duration: 1.18 | Throughput: 221707.73 tokens/sec2024-04-23 11:34:52.182 | INFO     | __main__:main:131 - [Step 12/1121] | Loss: 1.6953125 | Duration: 13.08 | Throughput: 40094.50 tokens/sec
...

Training Done
Saving Model...
Model saved in ./llama3_summarization
```

Upon checking the training logs, you can confirm that the training is progressing smoothly.

The throughput displayed during training indicates how many tokens per second the script is training through this PyTorch script.

- When using AMD MI250 GPU with 16 GPUs: Approximately 200,000 tokens/sec

The approximate training time depending on the GPU type and quantity is as follows:

- When using AMD MI250 GPU with 16 GPUs: Approximately 90 minutes

## Checking Accelerator Status During Training

During training, open another terminal and connect to the container. Then, execute the **`moreh-smi`** command to observe the MoAI Accelerator occupying memory and the training script running. Make sure to check this while the initialization process is completed and the training loss appears in the execution logs.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.2.0  Latest Version: 24.2.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Flavor     |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  4xLarge.2048GB  |  1806648 MiB   |  2096640 MiB   |    100%        |
+-----------------------------------------------------------------------------------------------------+
```