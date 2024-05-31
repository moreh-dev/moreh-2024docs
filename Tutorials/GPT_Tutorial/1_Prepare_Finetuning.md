---
icon: terminal
tags: [tutorial, gpt]
order: 40
---

# 1. Preparing for Fine-tuning

Preparing the PyTorch script execution environment on the MoAI Platform is similar to doing so on a typical GPU server.<br>
For a smooth tutorial experience, the following specifications are recommended:

- CPU: 16 cores or more

- Memory: 256GB or more

- MAF version: 24.5.0

- Storage: 100GB or more

Please verify that your environment meets these requirements before starting the tutorial.

## Checking PyTorch Installation

After connecting to the container via SSH, run the following command to check if PyTorch is installed in the current conda environment:

```bash
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.5.0          pypi_0    pypi
...
```

The version name includes both the PyTorch version and the version of MoAI required to run it.<br> In the example above, it indicates that version 24.5.0 of MoAI, which runs PyTorch version 1.13.1+cu116, is installed.

If you see the message **`conda: command not found`**, if the torch package is not listed, or if the torch package exists but does not include "moreh" in the version name, please follow the instructions in the  **[Prepare Fine-tuning on MoAI Platform](/Supported_Documents/Prepare_Fine_tuning_MoAI.md)** document to create a conda environment.

If the moreh version is not 24.5.0 but a different version, please execute the following code.

```bash
$ update-moreh --target 24.5.0
Currently installed: 24.3.0
Possible upgrading version: 24.5.0

Do you want to upgrade? (y/n, default:n)
y
```

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

## Downloading Training Script

Run the following command to download the PyTorch script for training from the GitHub repository.<br>
 In this tutorial, we will use the **`train_gpt.py`** script located inside the **`tutorial`** directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_gpt.py  ...
```

## Install Required Python Packages

Run the following command to install the third-party Python packages required to execute the script.

```bash
$ pip install -r requirements/requirements_gpt.txt
```
