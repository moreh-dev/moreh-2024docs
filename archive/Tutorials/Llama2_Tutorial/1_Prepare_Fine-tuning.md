---
icon: terminal
tags:  [tutorial, llama2]
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

If you see the message `conda: command not found`, if the torch package is not listed, or if the torch package exists but does not include "moreh" in the version name, please follow the instructions in the ***[Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md)*** document to create a conda environment.

After connecting to the container via SSH, run the following command to check if PyTorch is installed in the current conda environment:

```bash
$ update-moreh --target 24.2.0
Currently installed: 24.3.0
Possible upgrading version: 24.2.0

Do you want to upgrade? (y/n, default:n)
y
```

## Verifying PyTorch Installation

Run the following command to confirm that the torch package is properly imported and the MoAI Accelerator is recognized.

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


## Download the Training Script

Execute the following command to download the PyTorch script for training from the GitHub repository. In this tutorial, we will be using the **`train_llama2.py`** script located inside the **`tutorial`** directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_llama2.py  ...
```

## Install Required Python Packages

Execute the following command to install third-party Python packages required for script execution:

```bash
$ pip install -r requirements/requirements_baichuan.txt
```

## Download the Model and Tokenizer

Download the checkpoint and tokenizer for the Llama2-13b-hf model using Hugging Face. Please note that the Llama2 model requires community license agreement and Hugging Face token information. Additionally, since the checkpoint size for the Llama2 13B model is approximately 49GB, it is essential to have at least 50GB of storage space for the checkpoint.

Begin by visiting the following website and providing the required information to proceed with the license agreement.

[meta-llama/Llama-2-13b-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-13b-hf)

Once you've submitted the agreement form, check that the status on the page has updated as follows:

![](alert.png)

Once the status has changed, you can utilize the `download_llama2_13b.py` script found in the `tutorial` directory to download the model checkpoint and tokenizer into the `./llama-2-13b-hf directory.`

Make sure to replace `<user-token>` with your Hugging Face token.

```bash
~/quickstart$ python tutorial/download_llama2_13b.py --token <user-token>
```

Check if the model checkpoint and tokenizer have been downloaded.

```bash
~/quickstart$ ls ./llama-2-13b-hf
config.json                       model-00008-of-00011.safetensors
generation_config.json            model-00009-of-00011.safetensors
model-00001-of-00011.safetensors  model-00010-of-00011.safetensors
model-00002-of-00011.safetensors  model-00011-of-00011.safetensors
model-00003-of-00011.safetensors  model.safetensors.index.json
model-00004-of-00011.safetensors  special_tokens_map.json
model-00005-of-00011.safetensors  tokenizer_config.json
model-00006-of-00011.safetensors  tokenizer.json
model-00007-of-00011.safetensors  tokenizer.model
```

## Download Training Data

To download the training data, we'll use the **`prepare_llama2_dataset.py`** script located in the **`dataset`** directory. When you run the code, it will download the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset, preprocess it for training, and save it as **`llama2_dataset.pt`** file.

```bash
~/quickstart$ ls dataset
...  prepare_llama2_dataset.py ...

~/quickstart$ python dataset/prepare_llama2_dataset.py
2024-04-19 03:27:05,865 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpjkaqeu3r
2024-04-19 03:27:05,866 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpjkaqeu3r/_remote_module_non_scriptable.py
2024-04-19 03:27:24,010 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
Loading Tokenizer...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Downloading dataset...
Preprocessing dataset...
Saving datset into torch format...
Dataset saved as ./llama2_dataset.pt

~/quickstart$ ls
... llama2_dataset.pt ...
```

You can then load the stored dataset in your code like this:

```bash
dataset = torch.load("./llama2_dataset.pt")
```