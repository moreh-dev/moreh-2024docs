---
icon: terminal
tags: [tutorial, mistral]
order: 40
---
# 2. Understanding Training Code

Once you have prepared all the training data, let's delve into the contents of the **`train_mistral.py`** script to execute the actual fine-tuning process. In this step, you will confirm MoAI Platform's full compatibility with PyTorch, ensuring that the training code is identical to general PyTorch code for Nvidia GPUs. Moreover, you'll explore how efficiently MoAI Platform implements complex parallelization techniques beyond the conventional scope.

**We highly recommend proceeding with the tutorial using the provided script as is.** Afterward, feel free to customize the script to fine-tune the Llama2 13B model or any other publicly available model in a different manner. If needed, refer to the MoAI Platform application guide [**LLM Fine-tuning Parameter Guide**](/Supported_Documents/LLM_param_guide.md) provided by Moreh.


## Training Code

All the code used during training is exactly the same as when you're using PyTorch in general.

Import the necessary modules from the **`transformers`** library.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
```

Load the model configuration and checkpoint publicly available on Hugging Face. 

```python
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
```

Then load the training dataset from Hugging Face Hub, preprocess loaded dataset, and define the data loader.
In this tutorial, we will use the [python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) dataset available on Hugging Face among various datasets publicly available for code generation training.

```python
dataset = torch.load("iamtarun/python_code_instructions_18k_alpaca")
...
dataset = dataset.map(preprocess, num_proc=16)

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

With MoAI Platform, you can seamlessly use your existing PyTorch scripts without any modifications.

## About Advanced Parallelism

In the training script used in this tutorial, there is an additional line of code as follows, which executes the top-tier parallelization feature provided by the MoAI Platform:

```bash
torch.moreh.option.enable_advanced_parallelization()
```

For colossal language models like [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) used in this tutorial, it's imperative to train them using multiple GPUs. When using frameworks other than MoAI Platform, you'll need to introduce parallelization techniques such as Data Parallel, Pipeline Parallel, and Tensor Parallel.

For instance, if a user wants to apply DDP in their typical PyTorch code, they would need to add the following code snippet. (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)


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

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1") 
...
```

MoAI Platform's Advanced Parallelization(AP) provides optimization and automation features that are difficult to experience in other frameworks. Through the AP feature, users can experience the best distributed parallel processing. By leveraging AP, users can easily configure the optimal parameters and environment variables for Pipeline Parallelism and Tensor Parallelism required for training large-scale models with just a single line of code.
