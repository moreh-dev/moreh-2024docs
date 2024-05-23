---
icon: terminal
tags: [tutorial, qwen]
order: 700
---

# Qwen Fine-tuning

This tutorial introduces an example of fine-tuning the open-source [Qwen1.5 7B](https://huggingface.co/Qwen/Qwen1.5-7B) model on the MoAI Platform. Through this tutorial, you'll learn how to leverage the AMD GPU cluster using the MoAI Platform and explore the benefits of performance and automatic parallelization.

## Overview

The Qwen1.5 7B model is an open-source LLM released by [Tongyi Qianwen(通义千问)](https://www.alibabacloud.com/en/solutions/generative-ai/qwen?_p_lc=1) in China. In this tutorial, we'll be performing a code generation task on the MoAI Platform, fine-tuning the Qwen1.5 7B model using the  [python_code_instruction_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) dataset, which consists of system prompts, instructions for code generation, input values, and the code to be generated.

## Before You Start

Be sure to acquire a container or virtual machine on the MoAI Platform from your infrastructure provider and familiarize yourself with connecting to it via SSH. For example, you can sign up for the following public cloud service built on the MoAI Platform:
- KT Cloud’s Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

If you wish to temporarily allocate trial containers and GPU resources, please contact Moreh(support@moreh.io).

After connecting via SSH, run the **`moreh-smi`** command to ensure that the MoAI Accelerator is displayed correctly. The device name may vary depending on the system.


### Check MoAI Accelerator

To train models like the Llama2 model outlined in this tutorial, you need to select an appropriate size MoAI Accelerator. Start by using the **`moreh-smi`** command to check the currently used MoAI Accelerator.

Detailed instructions for selecting the MoAI Accelerator size required for the training will be provided in [**3. Model fine-tuning**](3_fine_tuning.md)


```jsx
$ moreh-smi
+---------------------------------------------------------------------------------------------------+
|                                                  Current Version: 24.2.0  Latest Version: 24.2.0  |
+---------------------------------------------------------------------------------------------------+
|  Device  |        Name         |      Model     |  Memory Usage  |  Total Memory  |  Utilization  |
+===================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+---------------------------------------------------------------------------------------------------+
```

