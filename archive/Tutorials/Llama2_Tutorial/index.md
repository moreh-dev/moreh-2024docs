---
icon: terminal
tags: [tutorial, llama2]
order: 1000
---

# Llama2 Fine-tuning

This tutorial introduces an example of fine-tuning the open-source [Llama2 13B model](https://huggingface.co/meta-llama/Llama-2-13b-hf) on the MoAI Platform. 
Through this tutorial, you'll learn how to leverage the AMD GPU cluster using the MoAI Platform and explore the benefits of performance and automatic parallelization.

### Overview

The Llama2 model, released by [Meta](https://about.meta.com/) in July 2023, is an open-source model based on the Decoder-only Transformer. It follows the structure of the existing Llama model but has been trained with 40% more data to understand more diverse and complex information.

Llama2 excels particularly in language understanding and generation tasks, achieving state-of-the-art performance in various natural language processing tasks. This model supports multilingual capabilities, enabling processing of text in various languages worldwide, and is publicly accessible for research and development purposes.

In this tutorial, we will fine-tune the Llama2 model on the MoAI Platform using the [CNN Daily Mail](https://huggingface.co/datasets/cnn_dailymail) dataset focus on summarization task. Summarization is one of the natural language processing techniques, where the task is to unravel long, complex text and deliver precise summaries.

### Before You Start

Be sure to acquire a container or virtual machine on the MoAI Platform from your infrastructure provider and familiarize yourself with connecting to it via SSH. For example, you can sign up for the following public cloud service built on the MoAI Platform:

- KT Cloud's Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

If you wish to temporarily allocate trial containers and GPU resources, please contact Moreh(support@moreh.io).

After connecting via SSH, run the `moreh-smi` command to ensure that the MoAI Accelerator is displayed correctly. The device name may vary depending on the system.

#### Check MoAI Accelerator 

To train models like the Llama2 model outlined in this tutorial, you need to select an appropriate size MoAI Accelerator. Start by using the `moreh-smi` command to check the currently used MoAI Accelerator.

Detailed instructions for selecting the MoAI Accelerator size required for the training will be provided in [3. Model fine-tuning](3_fine_tuning.md) 

```bash
$ moreh-smi
23:44:25 April 18, 2024
+---------------------------------------------------------------------------------------------------+
|                                                  Current Version: 24.2.0  Latest Version: 24.3.0  |
+---------------------------------------------------------------------------------------------------+
|  Device  |        Name         |      Model     |  Memory Usage  |  Total Memory  |  Utilization  |
+===================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+---------------------------------------------------------------------------------------------------+
```

