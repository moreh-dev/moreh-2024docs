---
icon: terminal
tags:  [tutorial, llama3_70b]
order: 40
---

# 2. Preparing for Fine-tuning

## Getting Started

To start, you'll need to obtain a container or virtual machine on the MoAI Platform from your infrastructure provider. You can use public cloud services based on the MoAI Platform, such as:

- Inquiries for MoAI Platform Trial Container: [support@moreh.io](https://www.notion.so/moreh/support@moreh.io)
- [KT Cloud Hyperscale AI Computing](https://cloud.kt.com/solution/hyperscaleAiComputing/)

After accessing the platform via SSH, run the **`moreh-smi`** command to ensure the MoAI Accelerator is properly recognized. Note that device names may vary depending on the system.

### Verifying the MoAI Accelerator

For this tutorial, which involves training a large-scale language model (LLM) like Llama3, selecting the appropriate size of MoAI Accelerator is crucial. First, use the **`moreh-smi`** command to check the current MoAI Accelerator in use.

Details on the specific MoAI Accelerator settings required for training will be provided in [**3. Model Fine-tuning**](3_fine_tuning.md)


```bash
$ moreh-smi
+---------------------------------------------------------------------------------------------------+
|                                                  Current Version: 24.5.0  Latest Version: 24.5.0  |
+---------------------------------------------------------------------------------------------------+
|  Device  |        Name         |      Model     |  Memory Usage  |  Total Memory  |  Utilization  |
+===================================================================================================+
|  * 0     |   MoAI Accelerator  |  xLarge.512GB  |  -             |  -             |  -            |
+---------------------------------------------------------------------------------------------------+
```

Setting up the PyTorch script execution environment on the MoAI Platform is similar to working on a standard GPU server.

## Checking PyTorch Installation

Once you’ve accessed the container via SSH, check if PyTorch is installed in the current conda environment by running:

```bash
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.5.0          pypi_0    pypi
...
```

The version should display both the PyTorch version and the MoAI version it’s running on. For instance, **`1.13.1+cu116`** indicates PyTorch version 1.13.1 with CUDA 11.6, and MoAI version 24.5.0.

If you see a **`conda: command not found`** message, the torch package isn’t listed, or the torch package doesn’t include "moreh" in its version name, follow the instructions in the [**Prepare Fine-tuning on MoAI Platform**](/Supported_Documents/Prepare_Fine_tuning_MoAI.md) document to create the conda environment.

### **Verifying PyTorch Functionality**

Run the following to ensure the torch package is properly imported and that the MoAI Accelerator is recognized:

```bash
$ python
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

## **Downloading the Training Script**

Download the PyTorch script for training from the GitHub repository by running:

For this tutorial, we will use the **`train_llama3.py`** script located in the **`tutorial`** directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_llama3.py  ...
```

## **Installing Required Python Packages**

Install third-party Python packages needed to run the script by executing:

```bash
$ pip install -r requirements/requirements_llama3.txt
```

## **Downloading the Model and Tokenizer**

Use Hugging Face to download the Llama3-70B model checkpoint and tokenizer. Note that using the Llama3 model requires agreeing to the community license and providing your Hugging Face token. Also, ensure you have about 150GB of storage space available, as the Llama3 70B model checkpoint is approximately 132GB.

First, enter the necessary information and agree to the license on the Hugging Face website.

[meta-llama/Llama-2-13b-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-13b-hf)

Once you've submitted the agreement form, check that the status on the page has updated as follows:

![](alert.png)

Once the status has changed, use the **`download_llama3_70b.py`** script in the **`tutorial`** directory to download the model checkpoint and tokenizer to the **`./llama3-70b`** directory:

Make sure to replace **`<user-token>`** with your Hugging Face token.

```bash
~/quickstart$ python tutorial/download_llama3_70b.py --token <user-token>
```

Verify that the model checkpoint and tokenizer have been downloaded.

```bash
~/quickstart$ ls ./llama3-70b
config.json              pytorch_model-00001-of-00015.bin  pytorch_model-00005-of-00015.bin  pytorch_model-00009-of-00015.bin  pytorch_model-00013-of-00015.bin  special_tokens_map.json
configuration_llama2.py  pytorch_model-00002-of-00015.bin  pytorch_model-00006-of-00015.bin  pytorch_model-00010-of-00015.bin  pytorch_model-00014-of-00015.bin  tokenizer_config.json
generation_config.json   pytorch_model-00003-of-00015.bin  pytorch_model-00007-of-00015.bin  pytorch_model-00011-of-00015.bin  pytorch_model-00015-of-00015.bin  tokenizer.json
modeling_llama.py        pytorch_model-00004-of-00015.bin  pytorch_model-00008-of-00015.bin  pytorch_model-00012-of-00015.bin  pytorch_model.bin.index.json
```
## **Downloading Training Data**

To download the training data, use the **`prepare_llama3_dataset.py`** script located in the **`dataset`** directory. Running this script will download and preprocess the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset, saving it as the **`llama3_dataset.pt`** file.

You can load the saved dataset in your code as follows:

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

Now you’re ready to proceed with the training process using the MoAI Platform.

```bash
dataset = torch.load("./llama3_dataset.pt")
```