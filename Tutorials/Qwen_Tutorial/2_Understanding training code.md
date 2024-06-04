---
icon: terminal
tags: [tutorial, qwen]
order: 40
---

# 2. Understanding Training Code

If you've prepared all the training data, let's now take a look at the **`train_qwen.py`** script to actually run the fine-tuning process. This script is a standard PyTorch code that executes fine-tuning based on the implementation of the Qwen model available in the Hugging Face Transformers library.

In this step, you'll observe that MoAI Platform is fully compatible with PyTorch, and the training code is exactly the same as standard PyTorch code designed for NVIDIA GPUs. **Moreover, you can also see how efficiently MoAI Platform can implement complex parallelization techniques.**

We recommend starting by using the provided script to complete the tutorial as-is. Afterwards, feel free to modify the script as desired to fine-tune the Qwen1.5 7B model in different ways. Thanks to MoAI Platform's full compatibility with PyTorch, such modifications are possible. If needed, refer to the [**LLM Fine-tuning parameter guide**](/Supported_Documents/LLM_param_guide.md) for assistance.


## Training Code

All the code used during training is exactly the same as the standard method of using PyTorch.

Firstly, import the required modules from the **`transformers`** library.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
```

Load the model configuration and checkpoint publicly available on Hugging Face. 

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B")
```

Then load the [training dataset](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) from Hugging Face Hub, preprocess loaded dataset, and define the data loader.

```python
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca").with_format("torch")
...
dataset = dataset.map(preprocess)
# Create a DataLoader for the training set
train_dataloader = torch.utils.data.DataLoader(
	dataset["train"],
	batch_size=args.batch_size,
	shuffle=True,
	drop_last=True,
)
```

Subsequently, the training proceeds similarly to general AI model training with Pytorch.

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

As shown above, you can code in the same way as traditional PyTorch code on MoAI Platform.

## About Advanced Parallelism

In the training script used in this tutorial, there is an additional line of code as follows, which executes the top-tier parallelization feature provided by the MoAI Platform:

```bash
torch.moreh.option.enable_advanced_parallelization()
```

To train massive language models like [Qwen1.5 7B](https://huggingface.co/Qwen/Qwen1.5-7B), which we use in this tutorial, it's inevitable to utilize multiple GPUs. If using different frameworks, you'll need to implement parallelization techniques such as Data Parallel, Pipeline Parallel, and Tensor Parallel to proceed with training.

For instance, if a user wants to apply DDP in a typical PyTorch code, the following code snippet would need to be added. (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

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

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B")
...
```

Experience optimal distributed parallel processing like no other framework can offer, thanks to MoAI Platform's Advanced Parallelization (AP), a feature that optimizes and automates parallelization in ways not found in other frameworks. With the AP feature, you can easily secure the optimal parameters and environment variables for Pipeline Parallelism and Tensor Parallelism, typically required for training large-scale models, with just one simple line of code.
