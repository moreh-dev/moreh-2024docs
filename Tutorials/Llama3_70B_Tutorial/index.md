---
icon: terminal
tags: [tutorial, llama3]
order: 1000
---

# Llama3 70B Fine-tuning

This tutorial introduces how to fine-tune the open-source [LLama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) model using the MoAI Platform. By following this tutorial, you will learn how to utilize AMD GPU clusters with the MoAI Platform and experience the benefits of performance optimization and automatic parallelization.

## MoAI Platform’s GPU Virtualization

The MoAI Platform is a scalable AI platform that allows for easy management of thousands of GPUs, facilitating the training and inference of AI models. One of the key features of the MoAI Platform is its ability to offer a simplified training process through virtualization and parallelization.

The MoAI Platform virtualizes multiple GPUs into a single entity known as the MoAI Accelerator. This approach makes it appear as though you are using just one GPU, eliminating the need for preparatory work or code modifications typically required for multi-GPU usage.


![](v_3.png)

When using a virtualized MoAI Accelerator, the platform automatically optimizes parallelization internally. By considering the model size and data volume, it provides the optimal parallelization strategy, allowing users to achieve high-performance training with minimal additional effort and straightforward code.