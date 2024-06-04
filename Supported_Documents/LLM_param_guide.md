---
icon: book
tags: [guide]
order: 10
---

# LLM Fine-tuning Parameter Guide


!!!primary 
This guide provides the optimal parameters recommended by the MoAI Platform and should be used as a reference during your training.
!!!

!!!secondary 
Please note that the names specified for MoAI Accelerators may vary depending on the Cloud Service Provider (CSP) you are using.
!!!

| Model | MoAI Platform version | MoAI Accelerator | Advanced Parallelism is applied | batch size | sequence length | vram Usage | Training Time | throughput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama3 8B | 24.5.0 | 2xlarge | True | 128 | 1024 | 867,021 MiB | 220m | 93,291 TPS |
| Llama3 8B | 24.5.0 | 4xlarge | True | 256 | 1024 | 1,366,564 MiB | 140m | 190,949 TPS |
| Llama3 8B | 24.5.0 | 8xlarge | True | 1024 | 1024 | 2,089,476 MiB | 78m | 394,605 TPS |
| Llama2 13B | 24.5.0 | 2xlarge | True | 128 | 1024 | 699,751 MiB | 560m | 78,274 TPS |
| Llama2 13B | 24.5.0 | 4xlarge | True | 256 | 1024 | 1,121,814 MiB | 249m | 150,406 TPS |
| Llama2 13B | 24.5.0 | 8xlarge | True | 512 | 1024 | 1,853,432 MiB | 144m | 315,004 TPS |
| Mistral 7B | 24.5.0 | 2xlarge | True | 256 | 1024 | 762652 MiB | 19m | 197,489 TPS |
| Mistral 7B | 24.5.0 | 4xlarge | True | 512 | 1024 | 1,147,841 MiB | 15m | 392,573 TPS |
| Mistral 7B | 24.5.0 | 8xlarge | True | 1024 | 1024 | 1,112,135 MiB | 16m | 798,760 TPS |
| Qwen1.5 7B | 24.5.0 | 2xlarge | True | 128 | 1024 | 758,555 MiB | 30m | 95302 TPS |
| Qwen1.5 7B | 24.5.0 | 4xlarge | True | 256 | 1024 | 1,403,640 MiB | 15m | 190,433 TPS |
| Qwen1.5 7B | 24.5.0 | 8xlarge | True | 512 | 1024 | 1,899,079 MiB | 14m | 381,714 TPS |
| Baichuan2 13B | 24.5.0 | 2xlarge | True | 128 | 1024 | 866,656 MiB | 30m | 99,873 TPS |
| Baichuan2 13B | 24.5.0 | 4xlarge | True | 256 | 1024 | 1,541,212 MiB | 28m | 191,605 TPS |
| Baichuan2 13B | 24.5.0 | 8xlarge | True | 512 | 1024 | 2,845,656 MiB | 17m | 384,165 TPS |
| Cerebras GPT 13B | 24.5.0 | 4xlarge | True | 16 | 1024 | 1,764,955 MiB | 81m | 6,841 TPS |
| Cerebras GPT 13B | 24.5.0 | 8xlarge | True | 32 | 1024 | 3,460,240 MiB | 62m | 13,286 TPS |
| Cerebras GPT 13B | 24.5.0 | 8xlarge | True | 16 | 2048 | 1,951,344 MiB | 100m | 18,001 TPS |
