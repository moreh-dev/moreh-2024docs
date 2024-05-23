---
icon: book
tags: [guide]
order: 90
---

# LLM Fine-tuning Parameter guide


!!!primary 
This guide provides optimal parameters offered by the MoAI Platform and should be used solely as a helpful reference during the training.
!!!

!!!secondary 
Keep in mind that the name of the MoAI Accelerator mentioned here might differ based on the CSP utilized by user.
!!!


| Model | MoAI Platform version | MoAI Accelerator | Advanced Parallelism is applied | batch size | sequence length |  Number of Tokens | VRAM Usage | Training Time | throughput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama2 13B | 24.3.0 | 2xlarge | True | 32 | 1024 |  | 543816 MiB | 872m | 8,934 |
| Llama2 13B | 24.3.0 | 2xlarge | True | 64 | 1024 |  | 749065 MiB | 586m | 11,562 |
| Llama2 13B | 24.3.0 | 2xlarge | True | 128 | 1024 |  | 790454 MiB | 400m | 65,565 |
| Llama2 13B | 24.3.0 | 4xlarge | True | 32 | 2048 |  | 1292886 MiB | 962m | 32,371 |
| Llama2 13B | 24.3.0 | 4xlarge | True | 64 | 2048 |  | 1600235 MiB | 720m | 63,893 |
| Llama2 13B | 24.3.0 | 4xlarge | True | 128 | 2048 |  | 1467646 MiB | 480m | 121,013 |
| Llama2 13B | 24.3.0 | 8xlarge | True | 32 | 4096 |  | 3181616 MiB | 1360m | 62,481 |
| Llama2 13B | 24.3.0 | 8xlarge | True | 64 | 4096 |  | 3143781 MiB | 720m | 125,180 |
| Llama2 13B | 24.3.0 | 8xlarge | True | 128 | 4096 |  | 3013826 MiB | 560m | 238,212 |
| Mistral 7B | 24.2.0 | 2xlarge | True | 256 | 1024 | 27B | 442,982 MiB | 59m | 15,972 |
| Mistral 7B | 24.2.0 | 2xlarge | True | 512 | 1024 |  | 560,835 MiB | 20m | 32,563 |
| Mistral 7B | 24.2.0 | 2xlarge | True | 1024 | 1024 |  | 790,572 MiB | 17m | 69,840 |
| Mistral 7B | 24.2.0 | 4xlarge | True | 256 | 2048 |  | 1,138,546 MiB | 25m | 62,740 |
| Mistral 7B | 24.2.0 | 4xlarge | True | 512 | 2048 |  | 1,138,546 MiB | 22m | 56,385 |
| Mistral 7B | 24.2.0 | 4xlarge | True | 1024 | 2048 |  | 1,138,546 MiB | 24m | 62,582 |
| Mistral 7B | 24.2.0 | 8xlarge | True | 256 | 4096 |  | 1,800,656 MiB | 36m | 157,859 |
| Mistral 7B | 24.2.0 | 8xlarge | True | 512 | 4096 |  | 1,800,656 MiB | 30m | 144,124 |
| Mistral 7B | 24.2.0 | 8xlarge | True | 1024 | 4096 |  | 1,767,888 MiB | 25m | 163,839 |
| Qwen1.5 7B | 24.5.0 | 2xlarge | True | 32 | 1024 |  | 626,391  MiB | 12m | 24,156 |
| Qwen1.5 7B | 24.5.0 | 2xlarge | True | 64 | 1024 |  | 784,485 MiB | 9m | 47,679 |
| Qwen1.5 7B | 24.5.0 | 2xlarge | True | 128 | 1024 |  | 638,460 MiB | 13m | 15,890 |
| Qwen1.5 7B | 24.5.0 | 4xlarge | True | 32 | 2048 |  | 1,403,047 MiB | 12m | 51,353 |
| Qwen1.5 7B | 24.5.0 | 4xlarge | True | 64 | 2048 |  | 1,122,745 MiB | 8m | 93,165 |
| Qwen1.5 7B | 24.5.0 | 4xlarge | True | 128 | 2048 |  | 1,680,233 MiB | 7m | 194,282 |
| Qwen1.5 7B | 24.5.0 | 8xlarge | True | 32 | 4096 |  | 1,706,797 MiB | 11m | 92,623 |
| Qwen1.5 7B | 24.5.0 | 8xlarge | True | 64 | 4096 |  | 1,651,008 MiB | 8m | 186,353 |
| Qwen1.5 7B | 24.5.0 | 8xlarge | True | 128 | 4096 |  | 2,146,115 MiB | 7m | 376,493 |
| Baichuan2 13B | 24.3.0 | 2xlarge | True | 32 | 1024 |  | 843,375 MiB | 40m | 26,111 |
| Baichuan2 13B | 24.3.0 | 2xlarge | True | 64 | 1024 |  | 858,128 MiB | 34m | 50,782 |
| Baichuan2 13B | 24.3.0 | 2xlarge | True | 128 | 1024 |  | 866,656 MiB | 30m | 99,873 |
| Baichuan2 13B | 24.3.0 | 4xlarge | True | 32 | 2048 |  | 1403,2189 | 38m | 58531 |
| Baichuan2 13B | 24.3.0 | 4xlarge | True | 64 | 2048 |  | 1489,3 | 35m | 109872 |
| Baichuan2 13B | 24.3.0 | 4xlarge | True | 128 | 2048 |  | 154,12123 | 28m | 191605 |
| Baichuan2 13B | 24.3.0 | 8xlarge | True | 32 | 4096 |  | 2,645,347 MiB | 22m | 172395 |
| Baichuan2 13B | 24.3.0 | 8xlarge | True | 64 | 4096 |  | 2,800,000 MiB | 20m | 172395 |
| Baichuan2 13B | 24.3.0 | 8xlarge | True | 128 | 4096 |  | 2,845,656 MiB | 17m | 172395 |
| Cerebras GPT 13B | 24.2.0 | 4xlarge | True | 16 | 1024 |  | 1,764,955 MiB | 81m | 6841 |
| Cerebras GPT 13B | 24.2.0 | 8xlarge | True | 32 | 1024 |  | 3,460,240 MiB | 62m | 13286 |
| Cerebras GPT 13B | 24.5.0 | 4xlarge | True | 16 | 2048 |  |  |  |  |
| Cerebras GPT 13B | 24.5.0 | 8xlarge | True | 16 | 2048 |  |  |  |  |
| Cerebras GPT 13B | 24.5.0 | 8xlarge | True | 32 | 2048 |  |  |  |  |

