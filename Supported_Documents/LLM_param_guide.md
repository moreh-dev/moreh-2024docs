---
icon: book
tags: [guide]
order: 10
---

# LLM Fine-tuning Parameter guide


!!!primary 
This guide provides the optimal parameters recommended by the MoAI Platform and should be used as a reference during your training.
!!!

!!!secondary 
Please note that the names specified for MoAI Accelerators may vary depending on the Cloud Service Provider (CSP) you are using.
!!!

| Model | MoAI Platform version | MoAI Accelerator | Advanced Parallelism is applied | batch size | sequence length | vram Usage | Training Time | throughput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama2 13B | 24.5.0 | 2xlarge | True | 128 | 1024 | 790454 MiB | 400m | 65,565 |
| Llama2 13B | 24.5.0 | 4xlarge | True | 256 | 1024 | 1,121,814 MiB | 233m | 150,406 |
| Llama2 13B | 24.5.0 | 8xlarge | True | 512 | 1024 | 1,853,432 MiB | 560m | 315,004 |
| Mistral 7B | 24.5.0 | 2xlarge | True | 1024 | 1024 | 790,572 MiB | 17m | 69,840 |
| Mistral 7B | 24.5.0 | 4xlarge | True | 1024 | 1024 | 1,138,546 MiB | 24m | 62,582 |
| Mistral 7B | 24.5.0 | 8xlarge | True | 1024 | 1024 | 1,767,888 MiB | 25m | 163,839 |
| Qwen1.5 7B | 24.5.0 | 2xlarge | True | 128 | 1024 | 638,460 MiB | 13m | 15,890 |
| Qwen1.5 7B | 24.5.0 | 4xlarge | True | 128 | 1024 | 1,680,233 MiB | 7m | 194,282 |
| Qwen1.5 7B | 24.5.0 | 8xlarge | True | 128 | 1024 | 2,146,115 MiB | 7m | 376,493 |
| Baichuan2 13B | 24.5.0 | 2xlarge | True | 128 | 1024 | 866,656 MiB | 30m | 99,873 |
| Baichuan2 13B | 24.5.0 | 4xlarge | True | 128 | 1024 | 154,12123 | 28m | 191,605 |
| Baichuan2 13B | 24.5.0 | 8xlarge | True | 128 | 1024 | 2,845,656 MiB | 17m | 384,165 |
| Cerebras GPT 13B | 24.5.0 | 4xlarge | True | 16 | 1024 | 1,764,955 MiB | 81m | 6,841 |
| Cerebras GPT 13B | 24.5.0 | 8xlarge | True | 32 | 1024 | 3,460,240 MiB | 62m | 13,286 |
| Cerebras GPT 13B | 24.5.0 | 8xlarge | True | 16 | 2048 | 1,951,344 MiB | 100m | 18,001 |
