---
icon: terminal
tags: [tutorial, baichuan]
order: 40
---

# 2. Understanding training code


If you've prepared all the training data, let's now take a look at the contents of `train_baichuan2_13b.py` script for the actual fine-tuning process. **In this step, you'll notice that MoAI Platform ensures full compatibility with PyTorch, confirming that the training code is identical to the typical PyTorch code for NVIDIA GPUs.** Additionally, you'll explore how efficiently MoAI Platform implements complex parallelization techniques beyond this.

**First and foremost, it's recommended to proceed with the tutorial using the provided script as is until the end.** Afterwards, you can modify the script as you wish to fine-tune the Baichuan model in different ways. If needed, refer to the [**LLM Fine-tuning Parameter Guide**](/Supported_Documents/LLM_param_guide.md).


# Training Code

**All the code is exactly the same as when using PyTorch conventionally.** 

First, import the necessary modules from the transformers library.

```python
from transformers import AutoModelForCausalLM, AdamW, AutoTokenizer
```

Load the model configuration and checkpoint from HuggingFace.

```python
model = AutoModelForCausalLM.from_pretrained('baichuan-inc/Baichuan-13B-Base', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('baichuan-inc/Baichuan-13B-Base', trust_remote_code=True)
```

Load the preprocessed dataset saved during the [**1. Preparing for Fine-tuning**](1_Prepare_Finetuning.md) and define the data loader.


```python
 dataset = torch.load('./baichuan_dataset.pt')

  # Create a DataLoader for the training set
  train_dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      drop_last=True,
  )
```

Subsequent training proceeds just like any other model training with PyTorch. 

```python
    # Mask pad tokens for training
    def mask_pads(input_ids, attention_mask, ignore_index = -100):
        idx_mask = attention_mask
        labels = copy.deepcopy(input_ids)
        labels[~idx_mask.bool()] = ignore_index
        return labels

    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=args.lr)

    # Start training
    for epoch in range(args.epoch):
        for i, batch in enumerate(train_dataloader, 0):
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            labels = mask_pads(input_ids, attn_mask)
            outputs = model(
                input_ids.cuda(),
                attention_mask=attn_mask.cuda(),
                labels=labels.cuda(),
                use_cache=False,
            )

            loss = outputs[0]
            loss.backward()

            optim.step()
            model.zero_grad(set_to_none=True)
```

**As shown above, with MoAI Platform, you can use your existing PyTorch scripts without any modifications.**

# About Advanced Parallelism

The training script used in this tutorial includes the following additional line of code, which performs automatic parallelization provided by MoAI Platform.

```bash
torch.moreh.option.enable_advanced_parallelization()
```

For huge language models like [Baichuan2 13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base), it's inevitable to train them using multiple GPUs. In such cases, if you're not using MoAI Platform, you'll need to introduce parallelization techniques like Data Parallel, Pipeline Parallel, and Tensor Parallelism.

For instance, if you want to apply DDP in your PyTorch code, you would need to add the following code snippet: (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

```python
...
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
...

def main(rank, world_size, args):
	setup(rank, world_size)
...
	sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
	loader = DataLoader(dataset, batch_size=64, sampler=sampler)
...

...
world_size = torch.cuda.device_count()  # Change this if you want a different number of GPUs
rank = int(os.environ['LOCAL_RANK'])
main(rank, world_size, args)
...
```

```bash
# Execute single node 
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py
# Execute multi node 
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train.py
```

DDP can be relatively easy to apply, but implementing techniques like [pipeline parallelism](https://pytorch.org/docs/stable/pipeline.html) or [tensor parallelism](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) requires quite complex code modifications. To apply optimized parallelization, you need to understand how Python code acts in a multiprocessing environment while writing the training scripts. Especially in multi-node setups, configuring the environment of each node used for training is necessary. Furthermore, finding the optimal parallelization method considering factors such as model type, size, and dataset requires a considerable amount of time.

**On the other hand, MoAI Platform's AP feature enables users to proceed with optimized parallelized training with just one line of code added to the training script, eliminating the need for users to manually apply additional parallelization techniques.**

```python
import torch
...
torch.moreh.option.enable_advanced_parallelization()

model = AutoModelForCausalLM.from_pretrained('baichuan-inc/Baichuan-13B-Base', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('baichuan-inc/Baichuan-13B-Base', trust_remote_code=True)
...
```

Experience the optimal automated distributed parallel processing that is only possible with MoAI Platform's Advanced Parallelization (AP) feature, unlike anything you've encountered in other frameworks. With AP, you can easily configure the optimal parameters and environment variables for pipeline parallelism and tensor parallelism, typically required for training large-scale models, with just a single line of code.
