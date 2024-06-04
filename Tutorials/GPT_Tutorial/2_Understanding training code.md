---
icon: terminal
tags: [tutorial, gpt]
order: 40
---
# 2. Understanding Training Code

Once you have prepared all the training data, let's take a look at the contents of the **`train_gpt.py`** script to execute the actual fine-tuning process. **In this step, you'll notice that the MoAI Platform offers full compatibility with PyTorch, meaning that the training code is 100% identical to typical PyTorch code for Nvidia GPUs.** **Furthermore, you'll see how efficiently the MoAI Platform implements complex parallelization techniques beyond what's traditionally possible.**

**We highly recommend proceeding with the tutorial using the provided script as is.** Afterward, feel free to customize the script to fine-tune the Cerebras-GPT-13B model or any other publicly available model in a different manner. If needed, refer to the MoAI Platform application guide [LLM Fine-tuning Parameter Guide](/Supported_Documents/LLM_param_guide.md).


## Training Code

**All code remains fully consistent with general PyTorch usage.**

Firstly, import the required modules from the **`transformers`** library.

```python
from transformers import AutoModelForCausalLM, AdamW, AutoTokenizer
```

Load the model configuration and checkpoint publicly available on Hugging Face. 

```python
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-13B")
tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B") 
```

Then load the training dataset from Hugging Face Hub, preprocess loaded dataset, and define the data loader.
In this tutorial, we will use the [Evol-Instruct-Python-26k](https://huggingface.co/datasets/mlabonne/Evol-Instruct-Python-26k) dataset. This dataset consists of Python code written in response to given prompt conditions.

```python
dataset = load_dataset("mlabonne/Evol-Instruct-Python-26k").with_format("torch")
...
dataset = dataset.map(preprocess)

# Create a DataLoader for the training set
train_dataloader = torch.utils.data.DataLoader(
	dataset,
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

For enormous language models like [Cerebras-GPT-13B](https://huggingface.co/cerebras/Cerebras-GPT-13B) used in this tutorial, it is inevitable to train them using multiple GPUs. In such cases, if you were to use frameworks other than the MoAI Platform, you would need to employ parallelization techniques like Data Parallel, Pipeline Parallel, or Tensor Parallel for training.

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

```python
import torch
...
torch.moreh.option.enable_advanced_parallelization()

model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-13B")
tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B") 
...
```

Experience optimal distributed parallel processing like no other framework can offer, thanks to MoAI Platform's Advanced Parallelization (AP), a feature that optimizes and automates parallelization in ways not found in other frameworks. With the AP feature, you can easily secure the optimal parameters and environment variables for Pipeline Parallelism and Tensor Parallelism, typically required for training large-scale models, with just one simple line of code.
