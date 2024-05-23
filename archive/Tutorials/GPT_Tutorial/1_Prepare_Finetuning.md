---
icon: terminal
tags: [tutorial, gpt]
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

If you see the message **`conda: command not found`**, if the torch package is not listed, or if the torch package exists but does not include "moreh" in the version name, please follow the instructions in the  **[Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md)** document to create a conda environment.


## Verifying PyTorch Installation

Run the following command to ensure that the torch package is imported correctly and the MoAI Accelerator is recognized.

```bash
$ python
Python 3.8.18 (default, Sep 11 2023, 13:40:15)
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

Execute the following command to pre-install third-party Python packages required for script execution:

```bash
$ pip install transformers==4.34.0 datasets==2.14.5 loguru==0.7.2
```

## Downloading Training Script

Run the following command to download the PyTorch script for training from the GitHub repository. In this tutorial, we will use the **`train_gpt.py`** script located inside the **`tutorial`** directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_gpt.py  ...
```

## Downloading Training Data

Hugging Face provides not only model checkpoints but also various datasets that can be used for model fine-tuning.

In this tutorial, we will use the [mlabonne/Evol-Instruct-Python-26k](https://huggingface.co/datasets/mlabonne/Evol-Instruct-Python-26k) dataset. This dataset consists of Python code written in response to given prompt conditions.

To download the training data, we will use the **`prepare_gpt_dataset.py`** script located in the **`dataset`** directory to download the dataset available on Hugging Face and preprocess it for immediate use in fine-tuning training.

```bash
~/quickstart$ python dataset/prepare_gpt_dataset.py
```

The preprocessed dataset is saved as **`gpt_dataset.pt`**.

The saved dataset can be loaded and used in code as follows.

```python
dataset = torch.load("gpt_dataset.pt")
```