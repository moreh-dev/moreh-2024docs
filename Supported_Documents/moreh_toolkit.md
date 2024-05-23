---
icon: tools
tags: [guide]
order: 110
---

# Moreh Toolkit Guide

The Moreh Toolkit is a command line tool designed to efficiently manage and monitor MoAI Accelerators on the MoAI Platform. With just three commands (**`moreh-smi`**, **`moreh-switch-model`**, **`update-moreh`**), users can effectively manage MoAI Accelerators and easily update MoAI Platform.

## Key Features

The main features of the Moreh Toolkit are as follows:

1. **Monitoring MoAI Accelerators:**
    - Use the **`moreh-smi`** command to monitor memory usage and process status in real-time.
2. **Switching AI Accelerators:**
    - Use the **`moreh-switch-model`** command to change AI accelerators and execute processes to achieve optimal performance.
3. **Updating and Rolling Back MoAI Platform:**
    - Use the **`update-moreh`** command to update Moreh solutions to the latest version or roll back to a previous version if needed.


## Monitoring MoAI Accelerators: `moreh-smi`

The **`moreh-smi`** command allows users to manage and monitor MoAI Accelerators. You can run it in a conda environment where MoAI Platform PyTorch is installed.

```jsx
$ moreh-smi
15:51:25 April 29, 2024 
+----------------------------------------------------------------------------------------------+
|                                                   Current Version:   Latest Version: 24.5.0  |
+----------------------------------------------------------------------------------------------+
|  Device  |        Name         |   Model   |  Memory Usage  |  Total Memory  |  Utilization  |
+==============================================================================================+
|  * 0     |  MoAI Accelerator   |  4xlarge  |  -             |  -             |  -            |
+----------------------------------------------------------------------------------------------+
```

If you are currently running training using the MoAI Accelerator, executing **`moreh-smi`** in a separate terminal session will display information about the running processes. Additionally, by using **`moreh-smi`**, you can quickly identify your Job ID. If you encounter any issues during training or inference on the MoAI Platform, providing your Job ID when contacting customer support will facilitate a faster response.

```jsx
$ moreh-smi
17:58:15 April 29, 2024 
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  xLarge.512GB    |  397 MiB       |  524160 MiB    |    0 %        |
+-----------------------------------------------------------------------------------------------------+

Processes:
+-----------------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID    |             Process            |  Memory Usage  |
+===================================================================================+
|       0  |  976356  |  1548305  |  python tutorial/train_gpt.py  |  397 MiB       |
+-----------------------------------------------------------------------------------+
```

#### Utilizing Multi-Accelerator Feature

By default, if users do not make any additional settings, there will be only one MoAI Accelerator in a single SSH environment. Typically, with one MoAI Accelerator, only one process can be executed at a time, allowing for only one process to run in a single SSH environment.

However, there may be scenarios where users want to leverage multiple MoAI Accelerators within the same SSH environment to run multiple processes simultaneously (e.g., running multiple training experiments concurrently with the same source code but different hyperparameters). In such cases, using **`moreh-smi`** to create multiple MoAI Accelerators within a single token enables running multiple processes concurrently.

Let's go through an example of adding, changing, and removing AI accelerators.

#### Adding AI Accelerators


First, let's add an AI accelerator. To use two or more AI accelerators, you can enter the **`moreh-smi device --add`** command, which will display the following interface.

```bash
(moreh) ubuntu@vm:~$ moreh-smi device --add
1. Small.64GB
2. Medium.128GB
3. Large.256GB
4. xLarge.512GB
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB
9. 6xLarge.3072GB
10. 8xLarge.4096GB
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB

Selection (1-13, q, Q):
```

Enter an integer corresponding to the model you want to use from 1 to 13, and a message "Create device success." will appear, indicating that the AI accelerator corresponding to the entered device number has been created. Up to 5 AI accelerators can be created within a single VM.

In the example below, let's add the [!badge variant="secondary" text=8xLarge.4096GB] AI accelerator with model number 10.

```bash
Selection (1-13, q, Q): 10
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |  AI Accelerator  |  Large.256GB        |
|    1     |  AI Accelerator  |  8xLarge.4096GB     |
+---------------------------------------------------+
Create device success.
1. Small.64GB
2. Medium.128GB
3. Large.256GB
4. xLarge.512GB
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB
9. 6xLarge.3072GB
10. 8xLarge.4096GB
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB

```

#### Changing Default AI Accelerator

The **`moreh-smi device --switch {Device_ID}`** command allows you to change the default MoAI Accelerator.

Here's how to use it:

```bash
(moreh) ubuntu@vm:~$ moreh-smi device --switch 1

+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|    0     |  KT AI Accelerator  |  2xLarge.1024GB  |
|  * 1     |  KT AI Accelerator  |  xLarge.512GB    |
|    2     |  KT AI Accelerator  |  2xLarge.1024GB  |
|    3     |  KT AI Accelerator  |  8xLarge.4096GB  |
|    4     |  KT AI Accelerator  |  Small.64GB      |
+---------------------------------------------------+
Switch Current Device success.
```

You can see that the default MoAI Accelerator has been changed to accelerator #1 in the example above.


```bash
Selection (0-4, q, Q): q

(moreh) ubuntu@vm:~$ moreh-smi
10:49:12 May 07, 2024
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.2.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |  KT AI Accelerator  |  2xLarge.1024GB  |  -             |  -             |  -            |
|  * 1     |  KT AI Accelerator  |  xLarge.512GB    |  -             |  -             |  -            |
|    2     |  KT AI Accelerator  |  2xLarge.1024GB  |  -             |  -             |  -            |
|    3     |  KT AI Accelerator  |  8xLarge.4096GB  |  -             |  -             |  -            |
|    4     |  KT AI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

## Changing AI Accelerators with `moreh-switch-model`

**`moreh-switch-model`** is a tool that allows you to change the flavor (specification) of the currently configured MoAI Accelerator. By changing the flavor of the MoAI Accelerator, you determine how much GPU memory to use.

Here's how to use it:

For example, if the result of the **`moreh-smi`** command is as follows, it means "The currently configured MoAI Accelerator is accelerator 0, and the type of this MoAI Accelerator is the **`Small.64GB`** model."

Here's how to use it:

```jsx
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
|    1     | MoAI Accelerator    |  Medium.128GB    |  -             |  -             |  -            |
|    2     | MoAI Accelerator    |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

When you use the **`moreh-switch-model`** command, an input prompt will appear like below:

```bash
(moreh) ubuntu@vm:~$ moreh-switch-model
Current AI Accelerator: Medium.128GB

1. Small.64GB  *
2. Medium.128GB  
3. Large.256GB
4. xLarge.512GB
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB
9. 6xLarge.3072GB
10. 8xLarge.4096GB
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB

Selection (1-13, q, Q):

```

Enter an integer (device number) corresponding to the model you want to use from 1 to 13, and the MoAI Accelerator will be switched to the MoAI Accelerator corresponding to the entered device number with the message "The MoAI Platform AI Accelerator model is successfully switched to {model_id}."

Let's change the MoAI Accelerator to the [!badge variant="secondary" text=Large.256GB] model with device number 3.

```bash
Selection (1-13, q, Q): 3
The AI Accelerator model is successfully switched to  "Large.256GB".

1. Small.64GB  
2. Medium.128GB
3. Large.256GB *
4. xLarge.512GB
5. 1.5xLarge.768GB
6. 2xLarge.1024GB
7. 3xLarge.1536GB
8. 4xLarge.2048GB
9. 6xLarge.3072GB
10. 8xLarge.4096GB
11. 12xLarge.6144GB
12. 24xLarge.12288GB
13. 48xLarge.24576GB

Selection (1-13, q, Q):

```

You can continue the change or exit the MoAI Accelerator change by entering **`q`** or **`Q`**.

After the change is complete, when you use **`moreh-smi`** again to check, the result will be as follows:


```jsx
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.2.0  Latest Version: 24.2.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     | MoAI Accelerator    |  Large.256GB     |  -             |  -             |  -            |
|    1     | MoAI Accelerator    |  Medium.128GB    |  -             |  -             |  -            |
|    2     | MoAI Accelerator    |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     | MoAI Accelerator    |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

You can see that the MoAI Accelerator type has been changed from the [!badge variant="secondary" text=Small.64GB] model to the [!badge variant="secondary" text=Large.256GB] model.

#### Removing AI Accelerators

Next, let's try deleting the accelerator corresponding to a specific device ID using the command **`moreh-smi device --rm {Device_ID}`**.

```bash
(moreh) ubuntu@vm:~$ moreh-smi --rm 1
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |  AI Accelerator  |    Large.256GB     |
+---------------------------------------------------+
Remove device success.
```

By entering the command above, the AI accelerator [!badge variant="secondary" text=8xLarge.4096GB] with Device ID 1 has been deleted. To confirm, run **`moreh-smi`** again to see that the device has been removed.

#### Utilizing Various Other Options

Besides, **`moreh-smi`** provides various other options. By using the **`--help`** option, you can see what options are available:

```jsx
$ moreh-smi --help

Usage: moreh-smi [-h | --help] [-r | --reset] [-s | --server-version] [-v | --version] [-t | --token] [-i | --idx]
                 [device {--add [model_id] | --rm [device_id] | --switch [device_id]}]

Basic Options:
  -h, --help             provide information about available command switches and their options
  -r, --reset            stop the running process
  -s, --server-version   print Moreh Framework version information
  -v, --version          print current software version information
  -t, --token            print Moreh Solution token information
  -i, --idx              select a device to print

Device Options:
  device --list                 list available models for adding device
  device --add [model_id]       add a device corresponding to model_id
  device --rm  [device_id]      remove a device corresponding to device_id
  device --switch [device_id]   switch to the device corresponding to [device_id]

Device Options operate interactively if there are no optional arguments([model_id], [device_id]).

Device Example:
  moreh-smi device --list
  moreh-smi device --add
  moreh-smi device --add 2
  moreh-smi device --switch 1
  moreh-smi -i 2
```

1. **`moreh-smi -p`** - Monitor detailed hardware status of MoAI Accelerators.
2. **`moreh-smi -t`** - Check MoAI Accelerator token information.
3. **`moreh-smi --reset`** - Terminate MoAI Accelerator processes.

## Updating MoAI Platform with `update-moreh`

**`update-moreh`** is a command that allows you to create a new conda environment and install Moreh solutions on it or update the version of Moreh solutions already installed in the conda environment. You can use **`update-moreh`** in the following situations:

- When you create a new conda environment, you need to install the required Python packages for Moreh solutions. In this case, you can easily install the latest version of Moreh solutions using the **`update-moreh`** command.

```jsx
$ conda create --name my_env python=3.8
$ update-moreh
```

- If you want to use the latest version of Moreh solutions in an already installed conda environment, you can update the currently installed Moreh solutions to the latest version using the **`update-moreh`** command alone.

```bash
$ update-moreh #update to latest version
```

- Sometimes, you may need to install a specific version of Moreh solutions. In this case, you can use the `--target` option to specify the specific version you want to install.

```bash
update-moreh --target 24.5.301 # Install version 24.5.301 
update-moreh --target 24.5.302 # Install version 24.5.302
```

- If MoAI Platform does not work properly due to dependency conflicts between different packages in the conda environment, you might have to reinstall the conda environment. In such cases, you can use **`update-moreh`** to restore the Moreh solutions in the conda environment. In the latter case, you can use the `--force` option to rebuild the environment. (Can be used with the `--target` option as well)

```bash
update-moreh --force --target 24.5.301
```

---