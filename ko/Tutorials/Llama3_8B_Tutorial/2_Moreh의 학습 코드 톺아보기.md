---
icon: terminal
tags: [guide]
order: 40
---
# 2. Moreh의 학습 코드 톺아보기

학습 데이터를 모두 준비하셨다면 다음으로는 실제 fine-tuning 과정을 실행할 `train_llama3.py` 스크립트의 내용에 대해 살펴 보겠습니다. 이 스크립트는 통상적인 PyTorch 코드로서 Hugging Face Transformers 라이브러리에 있는 Llama3 8B 모델 구현을 기반으로 fine tuning 작업을 실행합니다.

**우선 제공된 스크립트를 그대로 사용하여 튜토리얼을 끝까지 진행해 보시기를 권장합니다.** 이후 스크립트를 원하는 대로 수정하셔서 Llama2 13B 모델을 다른 방식으로 fine-tuning 하는 것도 얼마든지 가능합니다. MoAI Platform은 PyTorch와의 완전한 호환성을 제공하기 때문입니다. 필요하시다면 Moreh에서 제공하는 MoAI Platform 응용 가이드([LLM Fine-tuning 파라미터 가이드](/Supported_Documents/LLM_param_guide.md))를 참고하십시오.


## Training Code

**모든 코드는 일반적인 pytorch 사용 경험과 완벽하게 동일합니다.** 

먼저, `transformers` 라이브러리에서 필요한 모듈을 불러옵니다.

```python
from transformers import AdamW, LlamaForCausalLM, LlamaTokenizer
```

 앞서 다운로드 받았던 모델 체크포인트와 토크나이저를 불러옵니다.

```python
model = AutoModelForCausalLM.from_pretrained("./llama3-8b")
tokenizer = LlamaTokenizer.from_pretrained("./llama3-8b")
```

[1. Fine tuning 준비하기](1_Fine-tuning_준비하기.md) 단계에서 저장한 전처리된 데이터셋을 불러와 데이터로더를 정의합니다. 

```python
  dataset = torch.load("./llama3_dataset.pt")

  # Create a DataLoader for the training set
  train_dataloader = torch.utils.data.DataLoader(
      dataset["train"],
      batch_size=args.batch_size,
      shuffle=True,
      drop_last=True,
  )
```

이후 학습도 일반적인 Pytorch를 사용하여 모델 학습과 동일하게 진행됩니다. 

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
위와 같이 MoAI Platform에서는 기존 PyTorch 코드와 동일한 방식으로 작성하실 수 있습니다.


# About Advanced Parallelism

본 튜토리얼에 사용되는 학습 스크립트에서는 아래와 같은 코드가 추가로 한 줄 존재합니다. 이는 MoAI Platform에서 제공하는 최고의 병렬화 기능을 수행하는 코드입니다.

```bash
torch.moreh.option.enable_advanced_parallelization()
```
본 튜토리얼에서 사용하는 Llama3 8B와 같은 거대한 언어 모델의 경우 필연적으로 여러 개의 GPU를 사용하여 학습시켜야만 합니다. 이 경우 MoAI Platform이 아닌 다른 프레임워크를 사용할 경우, Data Parallel, Pipeline Parallel, Tensor Parallel과 같은 병렬화 기법을 도입하여 학습을 수행해야 합니다.

예를 들어, 사용자가 일반적인 pytorch 코드에서 DDP를 적용하고 싶다면, 다음과 같은 코드 스니펫이 추가되어야 합니다. 

[!ref https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)


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
# single node 실행
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py
# multi node 실행
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train.py
```

이와 같은 기본적인 설정 외에도 사용자는 학습 스크립트 작성 과정에서 Python 코드가 다중 처리(multi processing) 환경에서 어떻게 동작하는지 이해해야 하며, 특히 다중 노드(multi node) 설정에서는 학습에 사용되는 각 노드의 환경을 구성해야 합니다. 또한, 모델 종류, 크기, 데이터셋 등을 고려하여 최적의 병렬화 방법을 찾기 위해서는 상당한 시간이 필요합니다.

**반면, MoAI Platform의 AP 기능을 통해 사용자는 별도의 병렬화 기법을 적용할 필요 없이, 학습 스크립트에 단 한 줄의 코드를 추가하는 것으로도 최적화된 병렬화 학습을 진행할 수 있습니다.**


```bash
import torch
...
torch.moreh.option.enable_advanced_parallelization()

model = LlamaForCausalLM.from_pretrained("./llama-2-13b-hf")
...
```

이렇게 MoAI Platform의 Advanced Parallelization(AP)은 다른 프레임워크에서는 경험하기 어려운 최적화 및 자동화 기능을 제공합니다. AP 기능를 통해 **최적의 분산 병렬처리**를 경험해 보시기 바랍니다. AP기능을 이용하면 대규모 모델 훈련 시 필요한 Pipeline Parallelism, Tensor Parallelism의 최적 매개변수와 환경 변수 조합을 **아주 간단한 코드 한 줄**로 설정할 수 있습니다.