---
icon: terminal
tags: [tutorial, baichuan]
order: 600
expanded: false
---

# Baichuan2 Fine-tuning

The following tutorial will take you through the steps required to fine-tune [Baichuan2 13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base) model with an example dataset, using the MoAI Platform. Through the tutorial, you'll learn how to utilize an AMD GPU cluster with MoAI Platform and discover the benefits of improved performance and automatic parallelization.

## Overview

Baichuan2 is a large-scale multilingual language model developed by [Baichuan Intelligent Technology](https://github.com/baichuan-inc). This model offers configurations with 70 billion and 130 billion parameters trained on vast datasets consisting of 2.6 trillion tokens.

In this tutorial, we'll fine-tune the Baichuan2 13B model using the MoAI Platform with the [Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), a text-generation e-commerce dataset.

## Before You Start

Be sure to acquire a container or virtual machine on the MoAI Platform from your infrastructure provider and familiarize yourself with connecting to it via SSH. For example, you can sign up for the following public cloud service built on the MoAI Platform:

- KT Cloud’s Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

If you wish to temporarily allocate trial containers and GPU resources, please contact Moreh(support@moreh.io).


After connecting via SSH, run the `moreh-smi` command to ensure that the MoAI Accelerator is displayed correctly. The device name may vary depending on the system.

### Check MoAI Accelerator 

To train models like the Llama2 model outlined in this tutorial, you need to select an appropriate size MoAI Accelerator. Start by using the `moreh-smi` command to check the currently used MoAI Accelerator.

Detailed instructions for selecting the MoAI Accelerator size required for the training will be provided in [**3. Model fine-tuning**](3_finetuning.md).

```bash
$ moreh-smi
11:40:36 April 16, 2024
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.3.0  Latest Version: 24.3.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  Large.256GB  |  -             |  -             |  -            |
+-------------------------------------------------------------------------------------------------+
```

