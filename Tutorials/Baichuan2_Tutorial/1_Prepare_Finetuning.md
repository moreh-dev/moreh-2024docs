---
icon: terminal
tags: [tutorial, baichuan]
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
torch                     1.13.1+cu116.moreh24.3.0          pypi_0    pypi
...
```

The version name includes both the PyTorch version and the MoAI version required to execute it. In the example above, it indicates that PyTorch version 1.13.1+cu116 is running with MoAI version 24.3.0 installed.

If you encounter a `conda: command not found` message, or if the torch package is not listed, or if the torch package exists but does not include "moreh" in the version name, please follow the instructions in the **[Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md)** to create a conda environment.
If the moreh version is not 24.3.0 but a different version, please execute the following code.

```bash
$ update-moreh --target 24.3.0
Currently installed: 24.2.0
Possible upgrading version: 24.3.0

Do you want to upgrade? (y/n, default:n)
y
```


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

## Download the Training Script

Execute the following command to download the PyTorch script for training from the GitHub repository. In this tutorial, we will be using the `train_baichuan2_13b.py` script located inside the `tutorial` directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_baichuan2_13b.py  ...
```

## Install Required Python Packages

Execute the following command to install third-party Python packages required for script execution:

```bash
$ pip install -r requirements/requirements_baichuan.txt
```


## Download Training Data

Execute the following command to install third-party Python packages required for script execution:

To download the training data for this tutorial, we'll use the `prepare_baichuan_dataset.py` script located inside the `dataset` directory. When you run the code, it will download the [Bitext-custormer-support-llm-chatbot](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) dataset, and preprocess it for training, and save it as `baichuan_dataset.pt`.


```bash
~/quickstart$ ls dataset
...  prepare_baichuan_dataset.py ...

~/quickstart$ python dataset/prepare_baichuan_dataset.py
2024-04-19 03:27:05,865 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpjkaqeu3r
2024-04-19 03:27:05,866 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpjkaqeu3r/_remote_module_non_scriptable.py
2024-04-19 03:27:24,010 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
Loading Tokenizer...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Downloading dataset...
Preprocessing dataset...
Saving datset into torch format...
Dataset saved as ./baichuan_dataset.pt

~/quickstart$ ls
... baichuan_dataset.pt ...
```

The preprocessed dataset is saved as `baichuan_dataset.pt`.

Then, You can load the stored dataset in your code like this:

```Python
dataset = torch.load("baichuan_dataset.pt")
```