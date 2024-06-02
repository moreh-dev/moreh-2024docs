---
icon: terminal
tags: [tutorial, qwen]
order: 700
---

# Qwen Fine-tuning

This tutorial introduces an example of fine-tuning the open-source [Qwen1.5 7B](https://huggingface.co/Qwen/Qwen1.5-7B) model on the MoAI Platform. Through this tutorial, users can experience various features provided by the MoAI Platform and learn how to use an AMD GPU cluster.

- Users can easily run training without complex parallelization tasks or cluster environment setups, as they can treat dozens of GPUs as a single accelerator called MoAI Accelerator. This allows users to focus solely on training without worrying about resource management.
- Thanks to the automatic parallelization feature, code writing and development are simplified, and model training speed is significantly improved. This enables efficient resource utilization, allowing users to work faster and more effectively.

## Overview

he MoAI Platform is a scalable AI platform that enables easy control of thousands of GPUs for training and inference of AI models. One of its key features is providing a very simple training method through virtualization and parallelization when fine-tuning models.

The MoAI Platform provides multiple GPUs virtualized into a single accelerator called [MoAI Accelerator](https://docs.moreh.io/moai_features/virtualization/#gpu-virtualization-moai-accelerator). Therefore, there is no need for preprations or code modifications for using multiple GPUs.

The MoAI Platform automatically provides optimized parallelization when users use the virtualized MoAI Accelerator. It considers various parallelization methods based on model and data sizes to offer the optimal parallelization environment. As a result, users can experience high-performance training with simple code without any additional tasks.

## Getting Started

Please obtain a container or virtual machine on the MoAI Platform from your infrastructure provider and follow the instructions to connect via SSH. For example, you can apply for a trial container on the MoAI Platform or use public cloud services based on the MoAI Platform.

- MoAI Platform Trial Container (Inquiries: [support@moreh.io](mailto:support@moreh.io))
- KT Cloud's Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

After connecting via SSH, execute the **`moreh-smi`** command to verify that the MoAI Accelerator is properly detected. The device name may vary depending on the system.

### **Verifying MoAI Accelerator**

To train models like the sLLM introduced in this tutorial, it's important to select an appropriate size of the MoAI Accelerator. First, use the **`moreh-smi`** command to check the currently used MoAI Accelerator.

Detailed instructions for setting up the MoAI Accelerator specific to your training needs will be provided in the section [**3. Model fine-tuning**](3_fine_tuning.md)

```bash
$ moreh-smi
+---------------------------------------------------------------------------------------------------+
|                                                  Current Version: 24.2.0  Latest Version: 24.5.0  |
+---------------------------------------------------------------------------------------------------+
|  Device  |        Name         |      Model     |  Memory Usage  |  Total Memory  |  Utilization  |
+===================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+---------------------------------------------------------------------------------------------------+
```

