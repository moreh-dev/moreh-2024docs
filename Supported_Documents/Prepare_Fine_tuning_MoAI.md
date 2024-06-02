---
icon: terminal
tags: [guide]
order: 50
---

# Prepare Fine-tuning on MoAI Platform

The MoAI Platform can be configured with various GPUs, yet it provides a consistent user experience through a unified interface (CLI). This uniform access allows all users to interact with the system in the same way, making it more efficient and intuitive.

The MoAI Platform supports Python-based programming, similar to typical AI training environments. This document focuses on setting up and using a conda virtual environment as the standard configuration for AI training.

# Setting up a Conda Environment

1. To begin training, first create a conda environment:
    
    ```bash
    $ conda create --name <my-env> python=3.8
    ```
    
    Replace **`<my-env>`** with your desired environment name.
    
2. Activate the conda environment:
    
    ```bash
    $ conda activate <my-env>
    ```
    
3. The MoAI Platform supports various PyTorch versions, allowing you to choose the one that fits your needs.
    
    ```bash
    $ pip install torch==1.13.1+cu116.moreh24.5.0
    ```
    
4. Use the **`moreh-smi`** command to check the version of the installed Moreh solution and the details of the MoAI Accelerator in use. The current MoAI Accelerator is  [!badge variant="secondary" text="4xLarge.2048GB"]. For more information about the MoAI Accelerator, refer to the specifications.
    
    ```bash
    $ moreh-smi
    +-----------------------------------------------------------------------------------------------------+
    |                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
    +-----------------------------------------------------------------------------------------------------+
    |  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
    +=====================================================================================================+
    |  * 0     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
    +-----------------------------------------------------------------------------------------------------+
    ```
!!! 
For optimal parameters recommended for fine-tuning each model on the MoAI Platform, refer to the [LLM Fine-tuning parameter guide](/Supported_Documents/LLM_param_guide.md)
!!!


!!! 
For detailed usage of the moreh toolkit, including `moreh-smi` and `moreh-switch-model`, please refer to the Using the [MoAI Platform Toolkit ](/Supported_Documents/moreh_toolkit.md)
!!!



