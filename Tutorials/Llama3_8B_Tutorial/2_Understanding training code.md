---
icon: terminal
tags:  [tutorial, llama3]
order: 40
---
# 2. Understanding Training Code

Once you have prepared all the training data, let's take a look at the contents of the **`train_llama3.py`** script, which will carry out the actual fine-tuning process. This script executes fine-tuning based on the implementation of the Llama3 8B model in the Hugging Face Transformers library, using standard PyTorch code.

**We recommend initially proceeding with the provided script as is until the end of the tutorial.** Afterwards, feel free to modify the script as desired to fine-tune the Llama3 8B model in different ways. This flexibility is possible due to MoAI Platform's complete compatibility with PyTorch. If needed, refer to the MoAI Platform Application Guide provided by Moreh [LLM Fine-tuning Parameter Guide](/Supported_Documents/LLM_param_guide.md).


## Training Code

All the code used during training is exactly the same as when you're using PyTorch in general.

Import the necessary modules from the **`transformers`** library.

```python
from transformers import AdamW, LlamaForCausalLM, AutoTokenizer
```

Load the model configuration and checkpoint publicly available on Hugging Face. 

```python
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
```

Then load the [training dataset](https://huggingface.co/datasets/abisee/cnn_dailymail) from Hugging Face Hub, preprocess loaded dataset, and define the data loader.

```python
dataset = load_dataset("cnn_dailymail", "3.0.0").with_format("torch")
...
dataset = dataset.map(preprocess, num_proc=16)
# Create a DataLoader for the training set
train_dataloader = torch.utils.data.DataLoader(
	dataset["train"],
	batch_size=args.batch_size,
	shuffle=True,
	drop_last=True,
)
```

Subsequently, the training proceeds similarly to regular PyTorch model training.

```python
# Define AdamW optimizer
optim = AdamW(model.parameters(), lr=args.lr)

# Start training
for epoch in range(args.num_train_epochs):
	for step, batch in enumerate(train_dataloader, start=1):
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

In this way, you can write code in MoAI Platform using the same approach as with standard PyTorch code.

## About Advanced Parallelism

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

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
...
```

MoAI Platform's Advanced Parallelization (AP) provides optimization and automation features that are difficult to experience in other frameworks. 
With the AP feature, you can easily configure the optimal parameters and environment variables for Pipeline Parallelism and Tensor Parallelism which are typically required for large-scale model training, with **just a single line of code**.
