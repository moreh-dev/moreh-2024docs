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
Python 3.8.18 (default)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
...
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name()
[info] Requesting resources for MoAI Accelerator from the server...
[info] Initializing the worker daemon for MoAI Accelerator
[info] [1/1] Connecting to resources on the server (192.168.110.00:24158)...
[info] Establishing links to the resources...
[info] MoAI Accelerator is ready to use.
'MoAI Accelerator'
>>> quit()
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

## Install Required Python Packages

Run the following command to install the third-party Python packages required to execute the script.

```bash
$ pip install -r requirements/requirements_qwen.txt
```

## Download Training Data

To download the training data, we'll use the `prepare_qwen_dataset.py` script located in the **`dataset`** directory. When you run the code, it will download the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset, preprocess it for training, and save it as `qwen_dataset.pt` file.

```bash
~/quickstart$ ls dataset
...  prepare_qwen_dataset.py ...

~/quickstart$ python dataset/prepare_qwen_dataset.py
torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpjkaqeu3r
torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpjkaqeu3r/_remote_module_non_scriptable.py
datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
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
