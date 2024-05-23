---
icon: terminal
tags: [tutorial, mistral]
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

If you see the message conda: command not found, if the torch package is not listed, or if the torch package exists but does not include "moreh" in the version name, please follow the instructions in the **[Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md)** to create a conda environment.

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

Execute the following command to install third-party Python packages required for script execution:

```bash
$ pip install transformers==4.39.3 datasets==2.19.1 loguru==0.7.2
```

## Download the Training Script

Execute the following command to download the PyTorch script for training from the GitHub repository. In this tutorial, we will be using the `train_mistral.py` script located inside the **`tutorial`** directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_mistral.py  ...
```

## Download the Model and Tokenizer

Let's download the checkpoint and tokenizer for the Mistral 7B v0.1 model using Hugging Face. This process requires agreeing to the community license and providing your Hugging Face token information to access the Mistral model. Additionally, since the checkpoint size for the Mistral 7B model is approximately 15GB, it's essential to have at least 16GB of storage space available to store the checkpoint.

First, enter the required information on the Hugging Face website below and proceed with the license agreement.

[!ref icon="link-external" text="mistralai/Mistral-7B-v0.1 · Hugging Face"](https://huggingface.co/mistralai/Mistral-7B-v0.1)

After submitting the agreement form, confirm that the status on the page has changed as follows:

![](alert.png)

If the status has been updated, you can use the **`download_mistral_7b.py`** script located in the **`tutorial`** directory to download the model checkpoint and tokenizer into the **`./mistral-7b`** directory.

Replace **`<user-token>`** with your Hugging Face token.

```bash
~/quickstart$ python tutorial/download_mistral_7b.py --token <user-token>
```

Check if the model checkpoint and tokenizer have been downloaded.

```bash
~/quickstart$ ls ./mistral-7b
config.json                       model-00003-of-00006.safetensors  model.safetensors.index.json  tokenizer.model
generation_config.json            model-00004-of-00006.safetensors  special_tokens_map.json
model-00001-of-00006.safetensors  model-00005-of-00006.safetensors  tokenizer_config.json
model-00002-of-00006.safetensors  model-00006-of-00006.safetensors  tokenizer.json
```

## Download Training Data

In this tutorial, we will use the [python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) dataset (11.4 MB) available on Hugging Face among various datasets publicly available for code generation training.

We will execute **`prepare_mistral_dataset.py`** to download the dataset and preprocess it for training.

```
~/quickstart$ ls dataset
...  prepare_mistral_dataset.py ...

~/quickstart$ python dataset/dataset_qwen.py
2024-04-19 03:27:05,865 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpjkaqeu3r
2024-04-19 03:27:05,866 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpjkaqeu3r/_remote_module_non_scriptable.py
2024-04-19 03:27:24,010 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
Loading Tokenizer...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Downloading dataset...
Preprocessing dataset...
Saving datset into torch format...
Dataset saved as ./mistral_dataset.pt

~/quickstart$ ls
... mistral_dataset.pt ...
```


The preprocessed dataset will be saved as **`mistral_dataset.pt`**.

You can load the saved dataset in your code as follows.

```bash
dataset = torch.load("./mistral_dataset.pt")
```
