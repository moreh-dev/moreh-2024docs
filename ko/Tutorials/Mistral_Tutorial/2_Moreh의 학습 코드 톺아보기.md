---
icon: terminal
tags: [guide]
order: 40
---
# 2. Moreh의 학습 코드 톺아보기

학습 데이터를 모두 준비하셨다면 다음으로는 실제 fine-tuning 과정을 실행할 `train_mistral.py` 스크립트의 내용을 살펴 보겠습니다. **이번 단계에서는 MoAI Platform은 pytorch와의 완전한 호환성으로 학습 코드가 일반적인 nvidia gpu를 위한 pytorch 코드와 100% 동일하다는 것을 확인하실 수 있습니다.** **또한 이를 넘어서 기존의 복잡한 병렬화 기법들을 MoAI Platform에서는 얼마나 효율적으로 구현할 수 있는지도 확인하실 수 있습니다.**

**우선 제공된 스크립트를 그대로 사용하여 튜토리얼을 끝까지 진행해 보시기를 권장합니다.** 이후 스크립트를 원하는 대로 수정하셔서 Mistral 7B 모델, 혹은 다른 공개된 모델을 다른 방식으로 fine-tuning하는 것도 얼마든지 가능합니다. 필요하시다면 Moreh에서 제공하는 MoAI Platform 응용 가이드([LLM Fine-tuning 파라미터 가이드](/Supported_Documents/LLM_param_guide.md))를 참고하십시오.

## Training Code

**모든 코드는 일반적인 pytorch 사용 경험과 완벽하게 동일합니다.** 

먼저, `transformers` 라이브러리에서 필요한 모듈을 불러옵니다.

```python
from transformers import AutoModelForCausalLM, AdamW
```

HuggingFace에 공개된 모델 config와 체크포인트를 불러옵니다. 

```python
model = AutoModelForCausalLM.from_pretrained("./mistral-7b")
```

[Fine tuning 준비하기](1_Fine-tuning_준비하기.md) 단계에서 저장한 전처리된 데이터셋을 불러와 데이터로더를 정의합니다. 

```python
  dataset = torch.load("mistral_dataset.pt")

  # Create a DataLoader for the training set
  train_dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      drop_last=True,
  )
```

이후 학습도 일반적인 Pytorch를 사용하여 모델 학습과 동일하게 진행됩니다. 

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

**위와 같이 MoAI Platform에서는 기존 pytorch 코드와 동일한 방식으로 작성하실 수 있습니다.**

## About Advanced Parallelism

본 튜토리얼에 사용되는 학습 스크립트에서는 아래와 같은 코드가 추가로 한 줄 존재합니다. 이는 MoAI Platform에서 제공하는 최고의 병렬화 기능을 수행하는 코드입니다.

```bash
torch.moreh.option.enable_advanced_parallelization()
```

본 튜토리얼에서 사용하는 [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) 와 같은 거대한 언어 모델의 경우 필연적으로 여러 개의 GPU를 사용하여 학습시켜야만 합니다. 이때, MoAI Platform이 아닌 다른 프레임워크를 사용할 경우, Data Parallel, Pipeline Parallel , Tensor Parallel과 같은 병렬화 기법을 도입하여 학습을 수행해야 합니다. 

예를 들어, 사용자가 일반적인 pytorch 코드에서 DDP를 적용하고 싶다면, 다음과 같은 코드 스니펫이 추가되어야 합니다. (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

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

이와 같은 기본적인 세팅 이외에도 유저는 학습 스크립트 작성 과정에서 multi processing 환경에서의 Python 코드의 동작에 대해 이해하고 있어야 하며, 특히 multi node 세팅에서는 학습에 사용되는 노드들에 대한 환경 구성 작업이 추가적으로 들어가야 합니다. 게다가 모델의 종류, 크기, 데이터셋 등을 고려한 최적의 병렬화 방법을 찾기 위해서는 매우 많은 시간이 소요됩니다.

**반면, MoAI Platform의 AP기능은 유저가 직접 이러한 추가적인 병렬화 기법을 적용할 필요 없이, 단지 학습 스크립트에 다음과 같은 코드 한 줄을 추가하는 것 만으로도 최적화된 병렬화 학습을 진행할 수 있습니다.**

```python
import torch
...
torch.moreh.option.enable_advanced_parallelization()

model = AutoModelForCausalLM.from_pretrained("./mistral-7b")
tokenizer = AutoTokenizer.from_pretrained("./mistral-7b") 
...
```

이렇듯 다른 프레임워크에서는 경험할 수 없는 병렬화의 최적화 및 자동화 기능인 MoAI Platform만의 Advanced Parallelization(AP)을 통해 **최적의 분산 병렬처리**를 경험해보시기 바랍니다. AP기능을 이용하면 일반적으로 대규모 모델 훈련시 필요한 Pipeline Parallelism, Tensor Parallelism의 최적 매개변수와 환경변수 조합을 **아주 간단한 코드 한 줄**을 통해 확보할 수 있습니다.