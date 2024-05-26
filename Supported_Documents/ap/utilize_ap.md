---
icon: terminal
tags: [guide]
order: 69
expanded: false
---

# How to use AP

By default, AP(Advanced Parallelism) operates on a node-by-node basis. Therefore, a multi-GPU environment is required to use AP. Before proceeding with the AP feature, please review your current accelerator information using the guide below. For detailed information on accelerator sizes, refer to the [KT Hyperscale AI Computing (HAC) Service Accelerator Model Information](/Supported_Documents/KT_HAC_Models_Info.md).


### How to Apply AP

There are two ways to apply the AP feature:

1. Add a single line of code.
    
  실행 코드에 다음 한줄을 추가하여 AP 기능을 킬 수 있습니다. (이를 주석처리하면 끌 수 있습니다.)
    

```python
torch.moreh.option.enable_advanced_parallelization()
```

2. 환경 변수로 입력하기
    
  다음과 같이 터미널 세션의 환경변수로 AP 기능을 킬 수 있습니다. ( 0으로 설정하면 끌 수 있습니다.)
    

```bash
~/quickstart/ap-example$ MOREH_ENABLE_ADVANCED_PARALLELIZATION=1 python text_summarization_for_ap.py
```

### **Example Usage**

If you have an environment with two or more nodes ready, you can now create training code to use the AP feature. In this guide, we'll set up code using the Llama2 model. Note that the Llama2 model requires community license agreement and Hugging Face token information. Please refer to [1. Fine-tuning Preparation](/Tutorials/Llama2_Tutorial/1_Prepare_Fine-tuning.md) to prepare the training code.

Once the training code is ready, configure the PyTorch environment before running the training on the MoAI Platform. The example below shows the PyTorch 1.13.1+cu116 version running on MoAI Platform version 24.2.0. For detailed instructions, refer to the [1. Fine-tuning Preparation](/Tutorials/Llama2_Tutorial/1_Prepare_Fine-tuning.md) tutorial.

```bash
bashCopy code
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.2.0          pypi_0    pypi
...
```

Once the PyTorch environment is set up, fetch the training code from the GitHub repository.

```bash
bashCopy code
$ git clone https://github.com/moreh-dev/quickstart
$ cd quickstart
~/quickstart$ ls ap-example
... text_summarization_for_ap.py ...

```

Clone the **`quickstart`** repository and check the **`quickstart/ap-example`** directory. You'll find the **`text_summarization_for_ap.py`** file prepared by Moreh for testing the AP feature. Let's apply the AP feature using this code.

The training configuration for testing is as follows. We will proceed with testing based on this configuration.

- Batch Size: **`64`**
- Sequence Length: **`1024`**
- MoAI Accelerator: **`4xLarge`**


----

### Enabling the AP Feature (AP Feature ON)

프로그램의 main 함수 시작 지점에 AP 기능을 켜는 line이 있습니다. 다음과 같이 AP를 적용한 후 학습을 실행합니다.

At the beginning of the program's main function, there's a line to enable the AP feature. Apply AP and then run the training as shown below.

```python
pythonCopy code
def main(args):

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()
```

```bash
bashCopy code
~/quickstart$ python ap-example/text_summarization_for_ap.py
```

When the training starts, you will see logs like the following:

```bash
bashCopy code
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

As shown, by adding just one line to enable the AP feature, complex distributed parallel processing is executed, and training progresses. Next, we'll explain the scenario users might encounter if they do not use the AP feature.

### Disabling the AP Feature (AP Feature OFF)

Let's examine the situation when the AP feature is not used. To verify this, comment out the line that enables the AP feature at the beginning of the Python program's main function.

```python
pythonCopy code
def main(args):

    # Apply Advanced Parallelization
    # torch.moreh.option.enable_advanced_parallelization() # Commented out

```

Then proceed with the training.

```bash
bashCopy code
~/quickstart$ python ap-example/text_summarization_for_ap.py
```

After the training completes, you will see logs as the following.

```bash
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
Moreh solution has detected that the application requires more memory than what is currently available in at least one physical device of KT AI Accelerator.
>> Memory requested : 75051597828 bytes
>> Memory available : 68702699520 bytes
To address this issue, we recommend considering the following steps:
 1. Increase Device Size: If feasible, try increasing the size of the device, KT AI Accelerator, to accommodate the required memory.This can be done by using the `moreh-switch-model` command.
 2. Decrease Batch Size: Alternatively, you can decrease the batch size used in the application. By reducing the batch size by -b {new batch size} command, you can effectively manage the memory usage and ensure it fits within the available resources.
If the problem persists and you are unable to resolve it, please reach out to our technical support team for further assistance:
```

In the above logs, you can see the message **`RuntimeError: Error Code 4: OUT_OF_MEMORY`**, indicating an Out of Memory (OOM) error caused by trying to load data exceeding the VRAM of the 1 device chip, which is 64GB. 

If you were using a framework other than MoAI Platform, you would experience such inconvenience. However, as a user of the MoAI Platform, you can easily solve the troublesome OOM problem by applying the AP feature with just one line, without spending a long time calculating and deliberating separate parallelization optimizations.