---
icon: note
order: 108
---

# Automatic Parallelization

On the MoAI Platform, users interact with the MoAI Accelerator, which is presented as a single virtualized GPU. This means users can write code assuming the use of a single GPU without needing to worry about parallelization. But how does the MoAI Platform handle multiple GPUs?

**The MoAI Platform automatically optimizes and parallelizes based on the number of GPUs in use.**

For instance, if you start training with an accelerator flavor that uses 8 GPUs, the MoAI Platform automatically divides the total batch size into 8 parts, distributing them across each GPU. Let's take an example of fine-tuning the Llama3-8b model. If you select an accelerator flavor with 4 GPUs and set a batch size of 16, the platform automatically assigns 4 batches per GPU, processing approximately 125,000 tokens per second.

```bash
# Llama3-8b-base fine-tuning, batch-size 16, GPU 4
[Step 4/17944] | Loss: 2.03125 | Duration: 1.27 | Throughput: 12882.87 tokens/sec
[Step 6/17944] | Loss: 2.03125 | Duration: 1.22 | Throughput: 13393.38 tokens/sec
[Step 8/17944] | Loss: 2.109375 | Duration: 1.31 | Throughput: 12492.66 tokens/sec
[Step 10/17944] | Loss: 2.015625 | Duration: 1.24 | Throughput: 13201.98 tokens/sec
```

For faster and more efficient training, you can select an accelerator flavor with more GPUs and increase the batch size. If you choose an accelerator flavor with 16 GPUs and set the batch size to 64, the number of tokens processed per second can increase fourfold compared to the previous setup.


```bash
# Llama3-8b-base fine-tuning, batch-size 64, GPU 16
[Step 4/4486] | Loss: 2.125 | Duration: 1.42 | Throughput: 46148.86 tokens/sec
[Step 6/4486] | Loss: 2.078125 | Duration: 1.33 | Throughput: 49221.88 tokens/sec
[Step 8/4486] | Loss: 2.03125 | Duration: 1.33 | Throughput: 49392.99 tokens/sec
[Step 10/4486] | Loss: 2.046875 | Duration: 1.24 | Throughput: 52744.78 tokens/sec
```


But what if you want to use an even larger batch size? Without additional code modifications, a typical GPU cluster might run into Out of Memory (OOM) errors. However, the MoAI Platform can automatically parallelize the model to continue training.

When you choose an accelerator flavor with 16 GPUs and set the batch size to 512, the platform applies model parallelization and data parallelization simultaneously. This allows training with a larger batch size using the same number of GPUs.

```bash
# Llama3-8b-base fine-tuning, batch-size 512, gpu 16
[Step 4/560] | Loss: 1.953125 | Duration: 24.00 | Throughput: 21844.08 tokens/sec
[Step 6/560] | Loss: 1.8671875 | Duration: 24.63 | Throughput: 21283.67 tokens/sec
[Step 8/560] | Loss: 2.0 | Duration: 24.41 | Throughput: 21475.45 tokens/sec
[Step 10/560] | Loss: 1.9609375 | Duration: 24.26 | Throughput: 21609.36 tokens/sec
[Step 12/560] | Loss: 1.90625 | Duration: 24.43 | Throughput: 21463.95 tokens/sec
```


Additionally, even large models like the 70B model can be automatically parallelized without any extra work, making training straightforward. The MoAI Platform provides automatic optimization and parallelization based on the model and batch size, **enabling convenient and efficient use of multiple GPUs.**
