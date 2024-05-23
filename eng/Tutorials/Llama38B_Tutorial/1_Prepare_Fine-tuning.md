---
icon: terminal
tags:  [tutorial, llama3]
order: 40
---

# 1. Prepare Fine-tuning

Setting up the PyTorch execution environment on the MoAI Platform is similar to setting it up on a typical GPU server.

## Checking PyTorch Installation

After connecting to the container via SSH, run the following command to check if PyTorch is installed in the current conda environment:

```bash
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.2.0          pypi_0    pypi
...
```

The version name includes both the PyTorch version and the MoAI version required to run it. In the example above, it indicates that PyTorch 1.13.1+cu116 is installed with MoAI version 24.5.0.

If you see the message `conda: command not found`, if the torch package is not listed, or if the torch package exists but does not include "moreh" in the version name, please follow the instructions in the ***[Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md)*** document to create a conda environment.


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

Execute the following command to download the PyTorch script for training from the GitHub repository. In this tutorial, we will be using the **`train_llama3.py`** script located inside the **`tutorial`** directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_llama3.py  ...
```

## Install Required Python Packages

Execute the following command to install third-party Python packages required for script execution:

```bash
$ pip install -r requirements/requirements_llama3.txt
```

## Download the Model and Tokenizer

Use Hugging Face to download the checkpoint and tokenizer for the Llama3-8b model. Note that you will need to agree to the community license and provide your Hugging Face token information. Additionally, for the Llama3 8B model, you should have approximately 20GB of free storage available for the checkpoint, which is around 16GB.

First, enter the required information and agree to the license on the following site.

[https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

After submitting the agreement, confirm that the page status has changed as shown below.
![](alert.png)

Once the status has changed, you can use the **`download_llama3_8b.py`** script in the **`tutorial`** directory to download the model checkpoint and tokenizer to the **`./llama3-8b`** directory.

Replace **`<user-token>`** with your Hugging Face token.


```bash
~/quickstart$ python tutorial/download_llama3_8b.py --token <user-token>
```

Check if the model checkpoint and tokenizer have been downloaded successfully.

```bash
~/quickstart$ ls ./llama3-8b
config.json             model-00001-of-00004.safetensors  model-00003-of-00004.safetensors  model.safetensors.index.json  tokenizer_config.json
generation_config.json  model-00002-of-00004.safetensors  model-00004-of-00004.safetensors  special_tokens_map.json       tokenizer.json
```

## Download Training Data

To download the training data, use the **`prepare_llama3_dataset.py`** script in the **`dataset`** directory. Running this script will download and preprocess the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset for training, saving it as the **`llama3_dataset.pt`** file.

```bash
~/quickstart$ ls dataset
...  prepare_llama3_dataset.py ...

~/quickstart$ python dataset/prepare_llama3_dataset.py
2024-04-19 03:27:05,865 - torch.distributed.nn.jit.instantiator - INFO - Created a temporary directory at /tmp/tmpjkaqeu3r
2024-04-19 03:27:05,866 - torch.distributed.nn.jit.instantiator - INFO - Writing /tmp/tmpjkaqeu3r/_remote_module_non_scriptable.py
2024-04-19 03:27:24,010 - datasets - INFO - PyTorch version 1.13.1+cu116.moreh24.2.0 available.
Loading Tokenizer...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Downloading dataset...
Preprocessing dataset...
Saving datset into torch format...
Dataset saved as ./llama3_dataset.pt

~/quickstart$ ls
... llama3_dataset.pt ...
```

You can load and use the saved dataset in your code as follows.

```bash
dataset = torch.load("./llama3_dataset.pt")
```