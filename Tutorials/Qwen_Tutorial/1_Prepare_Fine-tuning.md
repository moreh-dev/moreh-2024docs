---
icon: terminal
tags: [tutorial, qwen]
order: 40
---

# 1. Prepare Fine-tuning

Preparing the PyTorch script execution environment on the MoAI Platform is similar to doing so on a typical GPU server.

## Checking PyTorch Installation

After connecting to the container via SSH, run the following command to check if PyTorch is installed in the current conda environment:


```bash
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.2.0          pypi_0    pypi
...
```
The version name includes both the PyTorch version and the version of MoAI required to run it. In the example above, it indicates that version 24.2.0 of MoAI, which runs PyTorch version 1.13.1+cu116, is installed.

If you see the message **`conda: command not found`**, if the torch package is not listed, or if the torch package exists but does not include "moreh" in the version name, please follow the instructions in the **[Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md)** document to create a conda environment.

## Verifying PyTorch Installation

Run the following command to ensure that the torch package is imported correctly and the MoAI Accelerator is recognized.

```bash
$ python
Python 3.8.19 (default, Sep 11 2023, 13:40:15)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
...
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name()
[2024-04-16 19:17:45.714] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-16 19:17:45.752] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-16 19:17:47.409] [info] [1/1] Connecting to resources on the server (192.168.110.00:24158)...
[2024-04-16 19:17:47.452] [info] Establishing links to the resources...
[2024-04-16 19:17:47.636] [info] MoAI Accelerator is ready to use.
'MoAI Accelerator'
>>> quit()
```

## Install Required Python Packages

Execute the following command to install third-party Python packages required for script execution:

```bash
$ pip install transformers==4.40.1 datasets==2.18.0 loguru==0.7.2 tiktoken==0.6.0
```

## Download the Training Script

Execute the following command to download the PyTorch script for training from the GitHub repository. In this tutorial, we will be using the `train_qwen.py` script located inside the `tutorial` directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_qwen.py  ...
```

## Download Training Data

To download the training data, we'll use the `prepare_qwen_dataset.py` script located in the **`dataset`** directory. When you run the code, it will download the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset, preprocess it for training, and save it as `qwen_dataset.pt` file.

```bash
~/quickstart$ ls dataset
...  prepare_qwen_dataset.py ...

~/quickstart$ python dataset/prepare_qwen_dataset.py
2024-04-19 03:27:05,865 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpjkaqeu3r
2024-04-19 03:27:05,866 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpjkaqeu3r/_remote_module_non_scriptable.py
2024-04-19 03:27:24,010 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
Loading Tokenizer...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Downloading dataset...
Preprocessing dataset...
Saving datset into torch format...
Dataset saved as ./qwen_dataset.pt

~/quickstart$ ls
... qwen_dataset.pt ...
```

The preprocessed dataset is saved as `qwen_dataset.pt`.

You can then load the stored dataset in your code like this:

```bash
dataset = torch.load("./qwen_dataset.pt")
```
