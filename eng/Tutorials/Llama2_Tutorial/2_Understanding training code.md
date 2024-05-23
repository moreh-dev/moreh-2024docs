---
icon: terminal
tags:  [tutorial, llama2]
order: 40
---
# 2. Understanding training code

If you've got all your training data ready, let's dive into running the actual fine-tuning process using the **`train_llama2.py`** script. This script is just standard PyTorch code, performing fine-tuning based on the Llama2 13B model from the Hugging Face Transformers library.

**We highly recommend proceeding with the tutorial using the provided script as is.** Afterward, feel free to customize the script to fine-tune the Llama2 13B model or any other publicly available model in a different manner. If needed, refer to the MoAI Platform application guide [**LLM Fine-tuning Parameter Guide**](/Supported_Documents/LLM_param_guide.md) provided by Moreh.

## Training Code

All the code used during training is exactly the same as when you're using PyTorch in general.

Import the necessary modules from the **`transformers`** library.

```python
from transformers import AdamW, LlamaForCausalLM, LlamaTokenizer
```

Then, load up the model checkpoint and tokenizer you downloaded earlier.

```python
model = AutoModelForCausalLM.from_pretrained("./llama-2-13b-hf")
tokenizer = LlamaTokenizer.from_pretrained("./llama-2-13b-hf")
```

Load your preprocessed dataset, which you prepared during the [1. Prepare fine-tuning](1_Prepare_Fine-tuning.md) step, and define your data loaders.


```python
  dataset = torch.load("./llama2_dataset.pt")

  # Create a DataLoader for the training set
  train_dataloader = torch.utils.data.DataLoader(
      dataset["train"],
      batch_size=args.batch_size,
      shuffle=True,
      drop_last=True,
  )
```

Training proceeds as usual, just like with any other PyTorch model.

```python
    # Compose pad token mask
    def create_mask(input_ids, tokenizer):
		    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
			  return (input_ids != pad_token_ids).long() 
			   
    # Mask pad tokens for training
    def mask_pads(inputs, tokenizer, ignore_index = -100):
        idx_mask = create_mask(inputs, tokenizer)
        labels = copy.deepcopy(inputs)
        labels[~idx_mask.bool()] = ignore_index
        return labels

    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=args.lr)

    # Start training
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader, start=1):
            #breakpoint()
            start_time = time.perf_counter()
            input_ids = batch["input_ids"]
            inputs, labels = input_ids, mask_pads(input_ids, tokenizer)
            attn_mask = create_mask(inputs, tokenizer)
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

**With MoAI Platform, you can seamlessly use your existing PyTorch scripts without any modifications.**

# About Advanced Parallelism

In the training script used in this tutorial, there is an additional line of code as follows, which executes the top-tier parallelization feature provided by the MoAI Platform:

```bash
torch.moreh.option.enable_advanced_parallelization()
```

Training a large language model like Llama2 13B requires a significant amount of GPUs. Without using the MoAI Platform, you would need to implement parallelization techniques such as data parallelism, pipeline parallelism, and tensor parallelism to perform the training.

(Reference: [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html))


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

While DDP can be relatively easy to apply, implementing techniques like [pipeline parallelism](https://pytorch.org/docs/stable/pipeline.html) or [tensor parallelism](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) involves quite complex code modifications. To apply optimized parallelization, you need to understand how Python code acts in a multiprocessing environment while writing the training scripts. Especially in multi-node setups, configuring the environment of each node used for training is necessary. Additionally, finding the optimal parallelization method considering factors such as model type, size, and dataset requires a considerable amount of time.

**In contrast, MoAI Platform's AP feature enables users to proceed with optimized parallelized training with just one line of code added to the training script, eliminating the need for users to manually apply additional parallelization techniques.**


```bash
import torch
...
torch.moreh.option.enable_advanced_parallelization()

model = LlamaForCausalLM.from_pretrained("./llama-2-13b-hf")
...
```

MoAI Platform's Advanced Parallelization (AP) provides optimization and automation features that are difficult to experience in other frameworks. 
With the AP feature, you can easily configure the optimal parameters and environment variables for Pipeline Parallelism and Tensor Parallelism which are typically required for large-scale model training, with **just a single line of code**.