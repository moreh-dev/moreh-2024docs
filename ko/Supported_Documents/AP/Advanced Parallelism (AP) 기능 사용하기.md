---
icon: terminal
tags: [ap]
order: 19
expanded: false
---

# AP 기능 사용하기

AP 기능은 노드 단위로 병렬화를 진행합니다. 따라서 AP를 사용하기 위해서는 멀티 노드 규모의 가속기 사용이 권장됩니다. AP 기능을 사용하기 전에 현재 사용하는 가속기 정보를 점검하시기 바랍니다.

## AP 기능 적용 방법

AP 기능은 다음과 같이 `import torch` 이후에  `torch.moreh..option.enable_advanced_parallelization()` 한 줄을 추가하여 적용할 수 있습니다.

```python
import torch

torch.moreh.option.enable_advanced_parallelization()
...
```

## 사용 예시 살펴보기

사용자가 2대 이상의 노드를 사용하는 환경이 준비 되었다면 이제 AP 기능을 사용하기 위한 학습 코드를 만들어 보겠습니다. 이 가이드에서는 Llama2 모델을 활용하여 코드를 세팅합니다. 참고로, Llama2 모델은 커뮤니티 라이센스 동의와 Hugging Face 토큰 정보가 필요합니다. [Llama2 1. Fine-tuning 준비하기](https://docs.moreh.io/ko/tutorials/llama2_13b_tutorial/1_fine-tuning_%EC%A4%80%EB%B9%84%ED%95%98%EA%B8%B0/) 를 참고하여 학습 코드를 준비해주세요. 

학습 코드가 준비되었다면, MoAI Platform에서 학습을 실행하기 전 아래와 같이 pytorch 환경을 설정합니다. 아래 예시의 경우 PyTorch 1.13.1+cu116 버전을 실행하는 MoAI Platform의 24.2.0 버전이 설치되어 있음을 의미합니다.

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

테스트를 위한 학습 구성은 다음과 같습니다. 이를 토대로 테스트를 진행하겠습니다.

- Batch Size: `64`
- Sequence Length: `1024`
- MoAI Accelerator: `4xLarge`

## AP 기능 ON

프로그램의 main 함수 시작 지점에 AP 기능을 켜는 line이 있습니다. 다음과 같이 AP를 적용한 후 학습을 실행합니다.

```python
def main(args):

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()  
```

```bash
~/quickstart$ python ap-example/text_summarization_for_ap.py
```

학습이 시작되면 다음과 같은 로그를 확인할 수 있습니다.

```bash

...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
[info] The number of candidates is 30.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 6103 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 7.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 808 [ms]
[info] SelectBestGraphFromCandidates finished.
info] Configuration for parallelism is selected.
[info] num_stages : 2, num_micro_batches : 4, batch_per_device : 1, No TP, recomputation : 0, distribute_param : true, distribute_low_prec_param : false
[info] train: true
|INFO     | __main__:main:151 - [Step 2/15] Loss: 1.6484375
|INFO     | __main__:main:151 - [Step 4/15] Loss: 1.828125
...
```

이처럼 단 한 줄의 AP 기능 프로그램을 추가하여 복잡한 분산 병렬처리가 수행되어 학습이 진행된 것을 확인할 수 있습니다. 다음은 사용자가 AP 기능을 사용하지 않을 경우 경험하게 될 상황을 가정하여 설명하겠습니다.

## AP 기능 OFF

AP 기능을 사용하지 않았을 때의 상황을 살펴보겠습니다.  이를 확인하기 위해, Python 프로그램의 main 함수 시작 지점에서 AP 기능을 켜는 줄을 주석 처리하여 AP 기능을 끄겠습니다.

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

```bash
...
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/4] Connecting to resources on the server (192.168.110.10:24163)...
[info] [2/4] Connecting to resources on the server (192.168.110.34:24163)...
[info] [3/4] Connecting to resources on the server (192.168.110.62:24163)...
[info] [4/4] Connecting to resources on the server (192.168.110.87:24163)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
Traceback (most recent call last):
  File "text_summarization_for_ap.py", line 183, in <module>
    main(args)
  File "text_summarization_for_ap.py", line 146, in main
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
RuntimeError: **Error Code 4: OUT_OF_MEMORY**
Moreh solution has detected that the application requires more memory than what is currently available in at least one physical device of MoAI Accelerator.
>> Memory requested : 75051597828 bytes
>> Memory available : 68702699520 bytes
To address this issue, we recommend considering the following steps:
 1. Increase Device Size: If feasible, try increasing the size of the device, MoAI Accelerator, to accommodate the required memory.This can be done by using the `moreh-switch-model` command.
 2. Decrease Batch Size: Alternatively, you can decrease the batch size used in the application. By reducing the batch size by -b {new batch size} command, you can effectively manage the memory usage and ensure it fits within the available resources.
If the problem persists and you are unable to resolve it, please reach out to our technical support team for further assistance:
```

위 로그에서 `RuntimeError: Error Code 4: OUT_OF_MEMORY` 라는 메시지를 볼 수 있습니다. 이것이 바로 앞서 말씀드린 1 device chip의 VRAM인 64GB를 초과하는 데이터를 로드할고 할 때 발생하는 OOM(Out Of Memory) 에러입니다. 

MoAI Platform 이 아닌 다른 프레임워크를 사용한다면 현재 사용 중인 GPU의 메모리 크기를 고려해 여러 병렬화 기법을 직접 적용하고 메모리 초과 이슈를 해결하기 위해 많은 시간과 노력을 들여 병렬화 및 최적화 작업을 직접 진행해야 합니다. 그러나 MoAI Platform을 사용하는 사용자는 AP 기능 한 줄만 추가하여 이러한 OOM 문제를 간단하게 해결할 수 있습니다.
