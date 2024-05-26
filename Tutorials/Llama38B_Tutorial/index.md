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

## Before You Start

Be sure to obtain a container or virtual machine on the MoAI Platform from your infrastructure provider and familiarize yourself with connecting to it via SSH. You can either request and use a trial container of the MoAI Platform or sign up for the public cloud service running on the MoAI Platform.

* Inquiries for MoAI Platform Trial Container: [support@moreh.io](support@moreh.io)

* [KT Cloud Hyperscale AI Computing](https://cloud.kt.com/solution/hyperscaleAiComputing/)

After connecting via SSH, run the **`moreh-smi`** command to ensure that the MoAI Accelerator is displayed correctly. The device name may vary depending on the system. 

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