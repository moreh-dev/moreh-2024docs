---
icon: terminal
tags:  [tutorial, llama3_70b]
order: 40
---

# 2. Preparing for Fine-tuning

## Getting Started

To start, you'll need to obtain a container or virtual machine on the MoAI Platform from your infrastructure provider. You can use public cloud services based on the MoAI Platform, such as:

- MoAI Platform Trial Container (Inquiries: [support@moreh.io](mailto:support@moreh.io))

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

### Verifying PyTorch Functionality

Run the following to ensure the torch package is properly imported and that the MoAI Accelerator is recognized:

```bash
$ python
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

## Downloading the Training Script

Download the PyTorch script for training from the GitHub repository by running:

For this tutorial, we will use the **`train_llama3.py`** script located in the **`tutorial`** directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_llama3.py  ...
```

## Installing Required Python Packages

Install third-party Python packages needed to run the script by executing:

```bash
$ pip install -r requirements/requirements_llama3.txt
```

## Acquire Access to the Model

To access and download the Llama3 70B model checkpoint from Hugging Face Hub, you will need to agree to the community license and provide your Hugging Face token information.

First, enter the necessary information and agree to the license on the Hugging Face website.

[!ref icon="link-external" text="meta-llama/Meta-Llama-3-70B · Hugging Face"](https://huggingface.co/meta-llama/Meta-Llama-3-70B)

Once you've submitted the agreement form, check that the status on the page has updated as follows:

![](alert.png)

Now you can authenticate your Hugging Face token with the following command:

```bash
huggingface-cli login
```
