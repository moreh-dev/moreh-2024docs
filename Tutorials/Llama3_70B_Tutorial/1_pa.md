---
icon: terminal
tags:  [tutorial, llama3_70b]
order: 40
---

# 1. MoAI Platform’s Parallelization - Llama3 70B

To fine-tune the Llama3 70B model, you must use multiple GPUs and implement parallelization techniques such as Tensor Parallelism, Pipeline Parallelism, and Data Parallelism. This typically involves using tools like Deepspeed and configuring complex settings. For pipeline parallelism, a deep understanding of the model is necessary. Additionally, finding the most effective parallelization method requires repeatedly modifying code and training the model to discover the optimal configuration. This process demands significant effort from users to utilize multiple GPUs effectively.

However, the MoAI Platform simplifies this by providing automatic parallelization and optimal configurations without the need for extensive code modifications or in-depth model knowledge, allowing users to train models effortlessly.

## Fine-tuning Code

**All code on the MoAI Platform operates the same way as standard PyTorch.** You can write scripts for training Llama3 70B as if you were using a single GPU in PyTorch.

```python
torch.moreh.option.enable_advanced_parallelization()
    
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
    
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

Unlike standard PyTorch, the MoAI Platform requires one additional line of code to enable automatic optimization and parallelization. This makes fine-tuning Llama3 70B straightforward.

```python
torch.moreh.option.enable_advanced_parallelization()
```

Now, let’s start the fine-tuning tutorial for the Llama3 70B model using 64 MI250 GPUs on the MoAI Platform. Through this tutorial, you will see how easy and effective it is to use multiple GPUs on the MoAI Platform.
