---
icon: terminal
tags: [guide]
order: 50
expanded: false
---

# Advanced Parallelism(AP)

By default, AP(Advanced Parallelism) operates on a node-by-node basis. Therefore, a multi-GPU environment is required to use AP. Before proceeding with the AP feature, please review your current accelerator information using the guide below. For detailed information on accelerator sizes, refer to the [KT Hyperscale AI Computing (HAC) Service Accelerator Model Information](/Supported_Documents/KT_HAC_Models_Info.md).


### How to Apply AP

There are two ways to apply the AP feature:

1. Add a single line of code.
    
  실행 코드에 다음 한줄을 추가하여 AP 기능을 킬 수 있습니다. (이를 주석처리하면 끌 수 있습니다.)
    

```python
torch.moreh.option.enable_advanced_parallelization()
```

1. 환경 변수로 입력하기
    
  다음과 같이 터미널 세션의 환경변수로 AP 기능을 킬 수 있습니다. ( 0으로 설정하면 끌 수 있습니다.)
    

```bash
~/quickstart/ap-example$ MOREH_ENABLE_ADVANCED_PARALLELIZATION=1 python text_summarization_for_ap.py
```

### 사용 예시 살펴보기

사용자가 2대 이상의 노드를 사용하는 환경이 준비 되었다면 이제 AP 기능을 사용하기 위한 학습 코드를 만들어 보겠습니다. 이 가이드에서는 Llama2 모델을 활용하여 코드를 세팅합니다. 참고로, Llama2 모델은 커뮤니티 라이센스 동의와 Hugging Face 토큰 정보가 필요합니다. [1. Fine-tuning 준비하기](https://www.notion.so/1-Fine-tuning-052d303ba7f34b5f89692af10bb53fd7?pvs=21) 를 참고하여 학습 코드를 준비해주세요. 

학습 코드가 준비되었다면, MoAI Platform에서 학습을 실행하기 전 아래와 같이 pytorch 환경을 설정합니다. 아래 예시의 경우 PyTorch 1.13.1+cu116 버전을 실행하는 MoAI의 24.2.0 버전이 설치되어 있음을 의미합니다. 자세한 설명은 [1. Fine-tuning 준비하기](https://www.notion.so/1-Fine-tuning-052d303ba7f34b5f89692af10bb53fd7?pvs=21) 튜토리얼을 참고해주시기 바랍니다.

```bash
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.2.0          pypi_0    pypi
...
```

Pytorch 환경 설정이 되었다면, Github 레포지토리에서 학습을 위한 코드를 가져옵니다.

```bash
$ git clone https://github.com/moreh-dev/quickstart
$ cd quickstart
~/quickstart$ ls ap-example
... text_summarization_for_ap.py ...
```

`quickstart` 레포지토리를 클론하여 `quickstart/ap-example` 디렉토리를 확인해보시면 Moreh에서 미리 준비한 AP기능 test를 위한 `text_summarization_for_ap.py`를 확인하실 수 있습니다. 이 코드를 기반으로 AP 기능을 적용해봅시다.

**text_summarization_for_ap.py** *(전체코드 제공)*

- 
    
    ```python
    import copy
    import torch
    
    from loguru import logger
    from datasets import load_dataset
    from argparse import ArgumentParser
    from transformers import AdamW, LlamaForCausalLM, LlamaTokenizer
    
    # Compose pad token mask
    def create_mask(input_ids, tokenizer):
        pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        return (input_ids != pad_token_ids).long()
    
    # Mask pad tokens
    def mask_pads(inputs, tokenizer, ignore_index = -100):
        idx_mask = create_mask(inputs, tokenizer)
        labels = copy.deepcopy(inputs)
        labels[~idx_mask.bool()] = ignore_index
        return labels
                          
    # Construct a formatted prompt
    def create_prompt(prompt):
        full_prompt = f"[SUMMARIZE] {prompt['article']} [/SUMMARIZE]\n{prompt['highlights']}"
        return full_prompt
    
    # Arguments
    def parse_args():
        parser = ArgumentParser(description="LLaMA2 FineTuning")
        parser.add_argument(
            "--model-name-or-path",
            type=str,
            help="model name or path",
        )
        parser.add_argument(
            "--num-train-epochs",
            type=int,
            default=1,
            help="num training epochs"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="train bacth size"
        )
        parser.add_argument(
            "--block-size",
            type=int,
            default=1024,
            help="max input token length"
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.00001,
            help="learning rate"
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=10,
            help="log interval"
        )
        parser.add_argument(
            "--save-model-dir",
            type=str,
            default="./outputs",
            help="path to save model"
        )
        args = parser.parse_args()
    
        return args
    
    def main(args):
    
        # Apply Advanced Parallelization
        torch.moreh.option.enable_advanced_parallelization()      
         
        # Load base model and tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
                                                                                                                                                                                                                                                                                                          # Set pad token
        tokenizer.pad_token_id = 0
    
        # Prepare the model for training on Accelerator
        model.cuda()
        model.train()
    
        # Load CNN/Daily Mail dataset and set its format to PyTorch tensors
        dataset = load_dataset("cnn_dailymail", '3.0.0').with_format("torch")
    
        # Tokenize and prepare the input prompt
        def preprocess(prompt):
            input_ids = tokenizer(
                create_prompt(prompt),
                return_attention_mask=False,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                max_length=args.block_size,
            )['input_ids']
            return {"input_ids": input_ids}
    
        # Apply preprocess function
        dataset = dataset.map(preprocess)
    
        # Create a DataLoader for the training set
        train_dataloader = torch.utils.data.DataLoader(
            dataset["train"],
            batch_size=args.batch_size,
            shuffle=True,
        )
    
        # Define AdamW optimizer
        optim = AdamW(model.parameters(), lr=args.lr)
    
        # Calculate total training steps
        total_step = len(train_dataloader) * args.num_train_epochs
    
        # Start training
        for step, batch in enumerate(train_dataloader, start=1):
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
            break
            model.zero_grad(set_to_none=True)
            if step % args.log_interval == 0:
                logger.info(f"[Step {step+(epoch*len(train_dataloader))}/{total_step}] Loss: {loss.item()}")
    
    		print("Training Done")
    		print("Saving Model...")
    		
        # Save trained model
        model = model.to("cpu")
        model.save_pretrained(args.save_model_dir)
        print(f"Model saved in '{args.save_model_dir}'")
        
    if __name__ == "__main__":
    
        args = parse_args()
        main(args)                                                                                                                                                                                                                                                                                           # Construct a formatted prompt                                                                                                                                                                                                                                                                    def create_prompt(prompt):                                                                                                                                                                                                                                                                            full_prompt = f"[SUMMARIZE] {prompt['article']} [/SUMMARIZE]\n{prompt['highlights']}"                                                                                                                                                                                                             return full_prompt 
    ```
    

테스트를 위한 학습 구성은 다음과 같습니다. 이를 토대로 테스트를 진행하겠습니다.

| num params | batch size | sequence length | sda |
| --- | --- | --- | --- |
| 13,015,864,320 | 64 | 1024 | 4xlarge |

먼저 AP를 적용시키는 부분이 어디인지 python 프로그램에서 확인해보시죠.

### AP 기능 ON

프로그램의 main 함수 시작 지점에 AP 기능을 켜는 line이 있습니다. 다음과 같이 AP를 적용한 후 학습을 실행합니다.

```python
def main(args):

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()  
```

```bash
~/quickstart$ python ap-example/text_summarization_for_ap.py
```

학습이 종료되면 다음과 같은 로그를 확인할 수 있습니다.

```bash
2024-04-15 14:05:17,959 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpcekf_lmo
2024-04-15 14:05:17,960 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpcekf_lmo/_remote_module_non_scriptable.py
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.15s/it]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 188.95 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:01<00:00, 207.51 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:01<00:00, 208.81 examples/s]
[2024-04-15 14:06:55.095] [info] Got DBs from backend for auto config.
[2024-04-15 14:06:57.473] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-15 14:06:57.490] [warning] A newer version of Moreh AI Framework is available. You can update the software to the latest version by running "update-moreh".
[2024-04-15 14:06:57.490] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-15 14:07:01.965] [info] [1/4] Connecting to resources on the server (192.168.110.20:24170)...
[2024-04-15 14:07:01.978] [info] [2/4] Connecting to resources on the server (192.168.110.21:24170)...
[2024-04-15 14:07:01.986] [info] [3/4] Connecting to resources on the server (192.168.110.45:24170)...
[2024-04-15 14:07:01.995] [info] [4/4] Connecting to resources on the server (192.168.110.99:24170)...
[2024-04-15 14:07:02.003] [info] Establishing links to the resources...
[2024-04-15 14:07:02.427] [info] MoAI Accelerator is ready to use.
[2024-04-15 14:07:02.815] [info] The number of candidates is 30.
[2024-04-15 14:07:02.815] [info] Parallel Graph Compile start...
[2024-04-15 14:07:08.919] [info] Elapsed Time to compile all candidates = 6103 [ms]
[2024-04-15 14:07:08.919] [info] Parallel Graph Compile finished.
[2024-04-15 14:07:08.919] [info] The number of possible candidates is 7.
[2024-04-15 14:07:08.919] [info] SelectBestGraphFromCandidates start...
[2024-04-15 14:07:09.728] [info] Elapsed Time to compute cost for survived candidates = 808 [ms]
[2024-04-15 14:07:09.729] [info] SelectBestGraphFromCandidates finished.
[2024-04-15 14:07:09.729] [info] Configuration for parallelism is selected.
[2024-04-15 14:07:09.729] [info] num_stages : 2, num_micro_batches : 4, batch_per_device : 1, No TP, recomputation : 0, distribute_param : true, distribute_low_prec_param : false
[2024-04-15 14:07:09.731] [info] train: true
2024-04-15 14:15:59.166 | INFO     | __main__:main:151 - [Step 2/15] Loss: 1.6484375
2024-04-15 14:16:08.663 | INFO     | __main__:main:151 - [Step 4/15] Loss: 1.828125
2024-04-15 14:16:18.561 | INFO     | __main__:main:151 - [Step 6/15] Loss: 1.671875
2024-04-15 14:16:28.356 | INFO     | __main__:main:151 - [Step 8/15] Loss: 1.6328125
2024-04-15 14:16:38.055 | INFO     | __main__:main:151 - [Step 10/15] Loss: 1.5703125
2024-04-15 14:16:47.711 | INFO     | __main__:main:151 - [Step 12/15] Loss: 1.640625
2024-04-15 14:16:57.388 | INFO     | __main__:main:151 - [Step 14/15] Loss: 1.6015625
Training Done
Saving Model...
Model saved in './outputs'
```

이처럼 단 한 줄의 AP 기능 프로그램을 추가하여 복잡한 분산 병렬처리가 수행되어 학습이 진행된 것을 확인할 수 있습니다. AP 기능을 적용하여 손쉬운 병렬화가 가능했는데요, 만약 사용자가 AP 기능을 사용하지 않았을 때는 어떤 경험을 하게 될까요? 

### AP 기능 OFF

이를 확인할 수 있도록 AP를 켜지 않았을 때의 형상을 보여 드리겠습니다. 다시 python 프로그램의 main 함수 시작 지점에 AP 기능을 켜는 line을 주석처리하여 AP 기능을 끄겠습니다. 

```python
def main(args):

    # Apply Advanced Parallelization
    # torch.moreh.option.enable_advanced_parallelization() # 주석처리
```

그 다음 학습을 진행합니다.

```bash
~/quickstart$ python ap-example/text_summarization_for_ap.py
```

학습이 종료되면 다음과 같은 로그를 확인할 수 있습니다. 

```json
2024-04-15 11:53:54,595 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpb4kvyiki
2024-04-15 11:53:54,595 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpb4kvyiki/_remote_module_non_scriptable.py
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.31s/it]
Downloading data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 8744.21it/s]
Extracting data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 1592.17it/s]
Generating train split: 100%|███████████████████████████████████████████████████████████████████████| 287113/287113 [00:04<00:00, 66267.80 examples/s]
Generating validation split: 100%|████████████████████████████████████████████████████████████████████| 13368/13368 [00:00<00:00, 76079.51 examples/s]
Generating test split: 100%|██████████████████████████████████████████████████████████████████████████| 11490/11490 [00:00<00:00, 74515.54 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 196.41 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:01<00:00, 211.83 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:01<00:00, 214.27 examples/s]
[2024-04-15 11:55:37.773] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-15 11:55:37.784] [warning] A newer version of Moreh AI Framework is available. You can update the software to the latest version by running "update-moreh".
[2024-04-15 11:55:37.784] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-15 11:55:42.446] [info] [1/4] Connecting to resources on the server (192.168.110.10:24163)...
[2024-04-15 11:55:42.456] [info] [2/4] Connecting to resources on the server (192.168.110.34:24163)...
[2024-04-15 11:55:42.463] [info] [3/4] Connecting to resources on the server (192.168.110.62:24163)...
[2024-04-15 11:55:42.470] [info] [4/4] Connecting to resources on the server (192.168.110.87:24163)...
[2024-04-15 11:55:42.478] [info] Establishing links to the resources...
[2024-04-15 11:55:42.907] [info] MoAI Accelerator is ready to use.
Traceback (most recent call last):
  File "text_summarization_2.py", line 183, in <module>
    main(args)
  File "text_summarization_2.py", line 146, in main
    optim.step()
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/optim/optimizer.py", line 140, in wrapper
    out = func(*args, **kwargs)
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/transformers/optimization.py", line 455, in step
    state["exp_avg"] = torch.zeros_like(p)
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/_M/driver/wrapper/moreh_wrapper.py", line 109, in wrapper
    raise instance
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/_M/driver/wrapper/moreh_wrapper.py", line 74, in wrapper
    return moreh_function(
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/_M/driver/builtin.py", line 15653, in zeros_like
    new_tensor = _make_filled_moreh_tensor_like('torch.zeros_like', None,
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/_M/driver/builtin.py", line 337, in _make_filled_moreh_tensor_like
    return _make_filled_moreh_tensor(
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/_M/driver/builtin.py", line 324, in _make_filled_moreh_tensor
    return frontend.register_operation_([new_tensor], op)[0]
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/_M/driver/common/frontend.py", line 773, in register_operation_
    return _register_operation_internal(input_tensors,
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/torch/_M/driver/common/frontend.py", line 641, in _register_operation_internal
    output_tickets = moreh_ir.create_operation(op_name, op.SerializeToString(),
RuntimeError: Error Code 4: OUT_OF_MEMORY
Moreh solution has detected that the application requires more memory than what is currently available in at least one physical device of KT AI Accelerator.
>> Memory requested : 75051597828 bytes
>> Memory available : 68702699520 bytes
To address this issue, we recommend considering the following steps:
 1. Increase Device Size: If feasible, try increasing the size of the device, KT AI Accelerator, to accommodate the required memory.This can be done by using the `moreh-switch-model` command.
 2. Decrease Batch Size: Alternatively, you can decrease the batch size used in the application. By reducing the batch size by -b {new batch size} command, you can effectively manage the memory usage and ensure it fits within the available resources.
If the problem persists and you are unable to resolve it, please reach out to our technical support team for further assistance:

error dump: 63 55 16 11 22 5 3 1 37 8 8 11 7 5 16 11 22 94 94 39 11 9 20 17 16 1 51 11 22 15 13 10 3 55 1 16 48 12 22 1 23 12 11 8 0 57 68 37 18 5 13 8 5 6 8 1 68 19 11 22 15 13 10 3 68 23 1 16 68 13 23 68 10 11 16 68 1 10 11 17 3 12 74
```

위 로그에서 `RuntimeError: Error Code 4: OUT_OF_MEMORY` 라는 메시지를 볼 수 있는데, 이것이 바로 앞서 말씀드린 1 device chip의 VRAM인 64GB가 넘는 데이터를 로드 할 수 없기 때문에 발생하는 OOM 에러입니다. 

MoAI Platform 이 아닌 다른 프레임워크를 사용한다면 이런 불편함을 겪어야 합니다. 그러나 MoAI Platform을 사용하는 사용자라면 별도 병렬화 최적화를 위해 오랫동안 계산하며 고민하는 시간을 들이지 않고 AP 기능 한줄을 적용하여 골치아픈 OOM 문제를 해결할 수 있습니다. 정말 편리한 기능이죠?