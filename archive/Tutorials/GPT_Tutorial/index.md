---
icon: terminal
tags:  [tutorial, gpt]
order: 800
---

# GPT Fine-tuning

This tutorial guides you on how to fine-tune GPT-based models open-sourced by [Hugging Face](https://huggingface.co/) on the MoAI Platform. Throughout this tutorial, you'll learn how to utilize an AMD GPU cluster with the MoAI Platform and explore the benefits of improved performance and automatic parallelization.

## Overview

GPT is a language model architecture that uses only the Transformer decoder structure. It was first introduced by [OpenAI](https://openai.com/) with GPT-1 in 2018. Since then, OpenAI has developed GPT-2, GPT-3, and GPT-4 models by increasing the dataset size and model parameters used for pre-training. Among them, the models that have been open-sourced are GPT-1 and GPT-2.

As the basic architecture of GPT is open-source, Hugging Face offers various GPT-based models beyond those developed by OpenAI.

In this tutorial, we'll use the MoAI Platform to fine-tune the [Cerebras-GPT-13B](https://huggingface.co/cerebras/Cerebras-GPT-13B) model for the code generation task.


## Before You Start

Make sure to obtain a container or virtual machine on the MoAI Platform from your infrastructure provider and learn how to connect to it via SSH. For instance, you can apply for the following public cloud service based on the MoAI Platform:

- KT Cloud's Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

If you wish to temporarily allocate trial containers and GPU resources, please contact Moreh(support@moreh.io).

After connecting via SSH, run the **`moreh-smi`** command to ensure that the MoAI Accelerator is displayed correctly. The device name may vary depending on the system. 


### Checking MoAI Accelerator

To train sLLMs like the GPT model we'll be guiding you through in this tutorial, you need to select an appropriate size MoAI Accelerator. First, use the **`moreh-smi`** command to check the currently used MoAI Accelerator.

Detailed instructions for selecting the MoAI Accelerator size required for the training will be provided in [**3. Finetuning Model**](3_finetuning.md)


```bash
$ moreh-smi
11:40:36 April 16, 2024
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.2.0  Latest Version: 24.2.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  Large.256GB  |  -             |  -             |  -           |
+-------------------------------------------------------------------------------------------------+
```

