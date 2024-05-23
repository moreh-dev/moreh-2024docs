---
icon: terminal
tags:  [tutorial, llama2]
order: 40
---

# 3. Model fine-tuning
Now, we will train the model through the following process. 

## Setting Accelerator Flavor
In MoAI Platform, physical GPUs are not directly exposed to users. Instead, virtual MoAI Accelerators are provided, which are available for use in PyTorch. By setting the accelerator's flavor, you can determine how much of the physical GPU will be utilized by PyTorch. Since the total training time and GPU usage cost vary depending on the selected accelerator flavor, users should make decisions based on their training scenarios. Refer to the following document to select the accelerator Flavor that aligns with your training objectives.

- **[KT Hyperscale AI Computing (HAC) AI Accelerator Information](/Supported_Documents/KT_HAC_Models_Info.md)**  
- **[LLM Fine-tuning Parameter Guide](/Supported_Documents/LLM_param_guide.md)**

!!!
Please refer to the document above or reach out to your infrastructure provider to inquire about the GPU types and quantities corresponding to each flavor.
!!!


Before continuing with the tutorial, we recommend reaching out to your infrastructure provider to inquire about the types and quantities of GPUs associated with each flavor. Once you have this information, you can choose one of the following flavors to proceed:

- AMD MI250 GPU with 16 units:
    - Select [!badge variant="secondary" text="4xlarge"] when using Moreh's trial container.
    - Select [!badge variant="secondary" text="4xLarge.2048GB"] when using KT Cloud's Hyperscale AI Computing.
- AMD MI210 GPU with 32 units.
- AMD MI300X GPU with 8 units.

Remember when we checked the MoAI Accelerator in the [1. Prepare Fine-tuning](1_Prepare_Fine-tuning.md)? Now let's set up the accelerator needed for learning.

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

Execute the given **`train_gpt.py`** script.

```bash
$ cd ~/quickstart
~/quickstart$ python tutorial/train_llama2.py
```

If the training proceeds smoothly, you should see the following logs. By going through this logs, you can verify that the Advanced Parallelism feature, which determines the optimal parallelization settings, is functioning properly. It's worth noting that, apart from the single line of AP code we looked at earlier in the PyTorch script, there is no handling for using multiple GPUs simultaneously in other parts of the script.

```bash
2024-04-24 13:51:32,884 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpypapv6uq
2024-04-24 13:51:32,886 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpypapv6uq/_remote_module_non_scriptable.py
2024-04-24 13:51:50,046 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
...
Loading checkpoint shards: 100%|██████████| 11/11 [00:11<00:00,  1.02s/it]
[2024-04-24 13:52:20.806] [info] Got DBs from backend for auto config.
[2024-04-24 13:52:25.469] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-24 13:52:25.481] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-24 13:52:30.967] [info] [1/4] Connecting to resources on the server (192.168.110.1:24159)...
[2024-04-24 13:52:30.982] [info] [2/4] Connecting to resources on the server (192.168.110.21:24159)...
[2024-04-24 13:52:30.989] [info] [3/4] Connecting to resources on the server (192.168.110.22:24159)...
[2024-04-24 13:52:30.995] [info] [4/4] Connecting to resources on the server (192.168.110.44:24159)...
[2024-04-24 13:52:31.035] [info] Establishing links to the resources...
[2024-04-24 13:52:31.906] [info] MoAI Accelerator is ready to use.
[2024-04-24 13:52:32.407] [info] The number of candidates is 22.
[2024-04-24 13:52:32.407] [info] Parallel Graph Compile start...
[2024-04-24 13:53:05.235] [info] Elapsed Time to compile all candidates = 32827 [ms]
[2024-04-24 13:53:05.235] [info] Parallel Graph Compile finished.
[2024-04-24 13:53:05.235] [info] The number of possible candidates is 4.
[2024-04-24 13:53:05.235] [info] SelectBestGraphFromCandidates start...
[2024-04-24 13:53:06.328] [info] Elapsed Time to compute cost for survived candidates = 1093 [ms]
[2024-04-24 13:53:06.328] [info] SelectBestGraphFromCandidates finished.
[2024-04-24 13:53:06.328] [info] Configuration for parallelism is selected.
[2024-04-24 13:53:06.328] [info] num_stages : 2, num_micro_batches : 8, batch_per_device : 1, No TP, recomputation : true, distribute_param : true
[2024-04-24 13:53:06.328] [info] train: true

2024-04-23 11:29:22.533 | INFO     | __main__:main:131 - [Step 2/1121] | Loss: 1.78125 | Duration: 16.31 | Throughput: 32150.92 tokens/sec
2024-04-23 11:30:29.368 | INFO     | __main__:main:131 - [Step 4/1121] | Loss: 1.7109375 | Duration: 15.65 | Throughput: 33494.69 tokens/sec
2024-04-23 11:31:36.496 | INFO     | __main__:main:131 - [Step 6/1121] | Loss: 1.75 | Duration: 15.68 | Throughput: 33444.54 tokens/sec
2024-04-23 11:32:40.688 | INFO     | __main__:main:131 - [Step 8/1121] | Loss: 1.609375 | Duration: 13.80 | Throughput: 37988.52 tokens/sec
2024-04-23 11:33:44.980 | INFO     | __main__:main:131 - [Step 10/1121] | Loss: 1.640625 | Duration: 16.25 | Throughput: 32272.10 tokens/sec
2024-04-23 11:34:52.182 | INFO     | __main__:main:131 - [Step 12/1121] | Loss: 1.6953125 | Duration: 13.08 | Throughput: 40094.50 tokens/sec
...

Training Done
Saving Model...
Model saved in ./llama2_summarization
```

You can verify that the training is proceeding smoothly by checking the training logs.

The throughput displayed during training indicates how many tokens are being trained per second through the PyTorch script.

- Throughput when using 16 AMD MI250 GPUs: Approximately 35,000 tokens/sec

Here are the approximate training times based on the type and number of GPUs:

- Training time when using 16 AMD MI250 GPUs: Approximately 10 hours

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