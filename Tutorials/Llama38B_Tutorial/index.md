---
icon: terminal
tags: [tutorial, llama3]
order: 1000
---

# Llama3 8B Fine-tuning

This tutorial introduces an example of fine-tuning the open-source [Llama3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B) model on the MoAI Platform. Through this tutorial, you will learn how to use an AMD GPU cluster with MoAI Platform and understand the benefits of its performance and automatic parallelization.

# Overview

The Llama3 model is an open-source, decoder-only Transformer model released by [Meta](https://about.meta.com/) in April 2024. It follows the architecture of previous Llama models but is trained on seven times more data (15T), enabling it to understand more diverse and complex information.

Llama3 excels in tasks involving language understanding and generation, achieving performance that significantly surpasses previous state-of-the-art results in various natural language processing tasks. It supports multiple languages, making it capable of processing texts from around the world, and is widely accessible for research and development purposes.

In this tutorial, we will fine-tune the Llama3 model on the MoAI Platform for a summarization task using the [CNN Daily Mail](https://huggingface.co/datasets/cnn_dailymail) dataset.

# **Getting Started**

Before you begin, ensure that you have access to a container or virtual machine on the MoAI Platform from an infrastructure provider, and that you have received instructions on how to connect via SSH. For instance, you can apply for and use the following public cloud service based on the MoAI Platform:

- Hyperscale AI Computing from KT Cloud (https://cloud.kt.com/solution/hyperscaleAiComputing/)

Alternatively, if you would like to temporarily allocate trial containers and GPU resources, please contact Moreh(support@moreh.io)

After connecting via SSH, run the **`moreh-smi`** command to verify that the MoAI Accelerator is properly displayed. Device names may vary depending on the system configuration.

### **Checking the MoAI Accelerator**

To train sLLMs like the Llama3 model described in this tutorial, you need to select an appropriately sized MoAI Accelerator. First, use the **`moreh-smi`** command to check the current MoAI Accelerator in use.

Detailed instructions on configuring the MoAI Accelerator for your specific training needs will be provided in section ["3. Model fine-tuning"](3_fine_tuning.md)

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