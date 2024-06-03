---
icon: terminal
tags: [guide]
order: 19
expanded: false
---

#  Utilizing AP

The AP feature enables parallelization at the node level. Therefore, it is recommended to use multi-node accelerators when using AP. Before using the AP feature, please check the information on the accelerators you are using.

### How to Apply the AP Feature

The AP feature can be applied by adding a single line of code after **`import torch`**:

```python
import torch

torch.moreh.option.enable_advanced_parallelization()
...
```

### Example Usage

If you have an environment with two or more nodes ready, you can now create training code to use the AP feature. In this guide, we'll set up code using the Llama2 model. Note that the Llama2 model requires community license agreement and Hugging Face token information. Please refer to [Llama2 Tutorial 1. Preparing for fine-tuning](https://docs.moreh.io/tutorials/llama2_tutorial/1_prepare_fine-tuning/) to prepare the training code.

Once the training code is ready, configure the PyTorch environment before running the training on the MoAI Platform. The example below shows the PyTorch 1.13.1+cu116 version running on MoAI Platform version 24.2.0.

```bash
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.2.0          pypi_0    pypi
...
```

Once the PyTorch environment is set up, fetch the training code from the GitHub repository.

```bash
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

### Enabling the AP Feature

At the beginning of the program's main function, there's a line to enable the AP feature. Apply AP and then run the training as shown below.

```python
def main(args):

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization()
```

```bash
~/quickstart$ python ap-example/text_summarization_for_ap.py
```

When the training starts, you will see logs like the following:

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

As shown, by adding just one line to enable the AP feature, complex distributed parallel processing is executed, and training progresses. Next, we'll explain the scenario users might encounter if they do not use the AP feature.

### Disabling the AP Feature

Let's examine the situation when the AP feature is not used. To verify this, comment out the line that enables the AP feature at the beginning of the Python program's main function.

```python
def main(args):

    # Apply Advanced Parallelization
    # torch.moreh.option.enable_advanced_parallelization() # Commented out

```

Then proceed with the training.

```bash
~/quickstart$ python ap-example/text_summarization_for_ap.py
```

After the training completes, you will see logs as the following.

```bash
...
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
Moreh solution has detected that the application requires more memory than what is currently available in at least one physical device of KT AI Accelerator.
>> Memory requested : 75051597828 bytes
>> Memory available : 68702699520 bytes
To address this issue, we recommend considering the following steps:
 1. Increase Device Size: If feasible, try increasing the size of the device, MoAI Accelerator, to accommodate the required memory.This can be done by using the `moreh-switch-model` command.
 2. Decrease Batch Size: Alternatively, you can decrease the batch size used in the application. By reducing the batch size by -b {new batch size} command, you can effectively manage the memory usage and ensure it fits within the available resources.
If the problem persists and you are unable to resolve it, please reach out to our technical support team for further assistance:
```

In the above logs, you can see the message **`RuntimeError: Error Code 4: OUT_OF_MEMORY`**, indicating an Out of Memory (OOM) error caused by trying to load data exceeding the VRAM of the 1 device chip, which is 64GB. 

If you were using a framework other than MoAI Platform, you would experience such inconvenience. However, as a user of the MoAI Platform, you can easily solve the troublesome OOM problem by applying the AP feature with just one line, without spending a long time calculating and deliberating separate parallelization optimizations.
