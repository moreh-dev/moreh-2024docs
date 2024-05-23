---
icon: terminal
tags: [guide]
order: 100
---

# Prepare Fine-tuning on MoAI Platform

The MoAI Platform can be customized with various GPUs while maintaining a consistent user experience via a command line interface(CLI). This uniform access ensures all users interact with the system in the same way, making it more efficient and intuitive.

Similar to general AI training environments, the MoAI Platform supports Python-based programming. This document focuses on setting up and using a conda virtual environment as the standard configuration for AI training.


## Setting up a Conda Environment

1. To begin training, first create a conda environment:
    
    ```bash
    $ conda create --name <my-env> python=3.8
    ```
    
    Replace `<my-env>` with your desired environment name.
    
2. Activate the conda environment:
    
    ```bash
    $ conda activate <my-env>
    ```
    
3. Install PyTorch. The MoAI Platform supports various PyTorch versions, allowing you to choose the one that fits your needs.
    
    ```bash
    $ pip install torch==1.13.1+cu116.moreh24.3.0
    $ pip install transformers==4.34.0
    $ pip install datasets==2.14.5
    $ pip install loguru==0.7.2
    ```
    
4. Use the `moreh-smi` command to check the version of the installed Moreh solution and the details of the MoAI Accelerator in use. The current MoAI Accelerator is [!badge variant="secondary" text=4xLarge.2048GB] For more information about the MoAI Accelerator, refer to the specifications.
    
    ```bash
    $ moreh-smi
    11:07:03 April 01, 2024
    +-----------------------------------------------------------------------------------------------------+
    |                                                    Current Version: 24.3.0  Latest Version: 24.3.0  |
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



