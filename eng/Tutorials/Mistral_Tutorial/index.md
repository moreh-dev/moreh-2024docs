---
icon: terminal
tags: [tutorial, mistral]
expanded: false
order: 900
---

# Mistral Fine-tuning

This tutorial guides you on fine-tuning the open-source [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) model on the MoAI Platform. You'll learn to utilize an AMD GPU cluster using the MoAI Platform and experience the improved performance and the benefits of automatic parallelization.

## Overview

The Mistral model, released by [Mistral AI](https://mistral.ai/) in 2023, is a giant language model. It has gained attention for outperforming larger models in complex tasks like code generation, question answering, and solving mathematical problems.

The Mistral 7B model uses only the Transformer's decoder, applying techniques like Sliding Window Attention to efficiently process the length of input tokens and introducing Rolling Buffer Cache to optimize memory usage.

In this tutorial, we'll fine-tune the Mistral 7B model using the [python_code_instructions_18k-alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) dataset for the code generation task on the MoAI Platform.

## Before You Start

Be sure to acquire a container or virtual machine on the MoAI Platform from your infrastructure provider and familiarize yourself with connecting to it via SSH. For example, you can sign up for the following public cloud service built on the MoAI Platform:

- KT Cloud’s Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

If you wish to temporarily allocate trial containers and GPU resources, please contact Moreh(support@moreh.io).

After connecting via SSH, run the **`moreh-smi`** command to ensure that the MoAI Accelerator is displayed correctly. The device name may vary depending on the system.

### Check MoAI Accelerator

To train models like the Llama2 model outlined in this tutorial, you need to select an appropriate size MoAI Accelerator. Start by using the **`moreh-smi`** command to check the currently used MoAI Accelerator.

Detailed instructions for selecting the MoAI Accelerator size required for the training will be provided in [3. Model fine-tuning](3_fine_tuning.md).

```bash
$ moreh-smi
11:40:36 April 16, 2024
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.2.0  Latest Version: 24.2.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  Small.64GB  |  -             |  -             |  -            |
+-------------------------------------------------------------------------------------------------+
```