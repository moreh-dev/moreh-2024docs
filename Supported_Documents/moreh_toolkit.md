---
icon: tools
tags: [guide, moreh_toolkit]
order: 60

---

# Moreh Toolkit Guide

The Moreh Toolkit is a command line tool designed to efficiently manage and monitor [MoAI Accelerator](https://docs.moreh.io/moai_features/virtualization/) on the MoAI Platform. With just three commands (**`moreh-smi`**, **`moreh-switch-model`**, **`update-moreh`**), users can effectively manage MoAI Accelerators and easily update MoAI Platform.

## Key Features

The main features of the Moreh Toolkit are as follows:

1. **Monitoring MoAI Accelerators:**
    - Use the **`moreh-smi`** command to monitor memory usage and process status in real-time.
2. **Switching AI Accelerators:**
    - Use the **`moreh-switch-model`** command to change AI accelerators and execute processes to achieve optimal performance.
3. **Updating and Rolling Back MoAI Platform:**
    - Use the **`update-moreh`** command to update MoAI Platform to the latest version or roll back to a previous version if needed.

## MoAI Accelerator Monitoring: `moreh-smi`

`moreh-smi` is a command-line tool that allows users to manage and monitor the MoAI Accelerator. You can run it in a conda environment where MoAI Platform PyTorch is installed.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  4xlarge.2048GB  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

If you are currently running a training session using the MoAI Accelerator, running `moreh-smi` in another terminal session will display the running process information as follows. You can also use `moreh-smi` to quickly identify your Job ID, allowing for faster support response from MoAI Platform in case of training or inference issues. In the example below, the Job ID is 976356.

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |  MoAI Accelerator   |  xLarge.512GB    |    397 MiB     |  524160 MiB    |      0 %      |
+-----------------------------------------------------------------------------------------------------+

Processes:
+-----------------------------------------------------------------------------------+
|  Device  |  Job ID  |    PID    |             Process            |  Memory Usage  |
+===================================================================================+
|       0  |  976356  |  1548305  |  python tutorial/train_gpt.py  |    397 MiB     |
+-----------------------------------------------------------------------------------+
```

### Utilizing MoAI Accelerator's Multi Accelerator Feature

By default, if users do not configure anything, there will only be one MoAI Accelerator in a VM or container environment. With one MoAI Accelerator, only one process can run. However, there may be cases where you want to run multiple processes concurrently in the same environment, even with a single MoAI Accelerator. For example, you may want to run multiple training experiments concurrently by changing the same source code or hyperparameters. In such cases, you can create multiple MoAI Accelerators within a single token using `moreh-smi`, enabling multiple processes to run concurrently.

Let's explore adding, modifying, and removing MoAI Accelerators with the following examples.

### Adding MoAI Accelerators

First, let's add a MoAI Accelerator. When you enter the `moreh-smi device --add` command to use two or more MoAI Accelerators, you will see the following interface.

```bash
$ moreh-smi device --add
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

When you input the integer corresponding to the model you want to use from 1 to 13, a MoAI Accelerator corresponding to the entered device number will be created with the message "Create device success." Within one environment, you can create a maximum of 5 AI accelerators. If you need to create more MoAI Accelerators, please contact your infrastructure administrator.


In the example below, let's add the 10th [!badge variant="secondary" text=8xLarge.4096GB] MoAI Accelerator:

```bash
$ moreh-smi device --add 10
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |   MoAI Accelerator  |  xLarge.512GB    |  -             |  -             |  -            |
|  * 1     |   MoAI Accelerator  |  8xLarge.4096GB  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

---

### Changing the Default MoAI Accelerator: `moreh-smi device --switch`

`moreh-smi device --switch {Device_ID}` is a command that allows you to change the default MoAI Accelerator.

It can be used as follows:

```bash
$ moreh-smi device --switch 1

+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|    0     |   MoAI Accelerator  |  2xLarge.1024GB  |
|  * 1     |   MoAI Accelerator  |  xLarge.512GB    |
|    2     |   MoAI Accelerator  |  2xLarge.1024GB  |
|    3     |   MoAI Accelerator  |  8xLarge.4096GB  |
|    4     |   MoAI Accelerator  |  Small.64GB      |
+---------------------------------------------------+
Switch Current Device success.
```

This means that the current default MoAI Accelerator has been changed to Accelerator 1.

```bash
Selection (0-4, q, Q): q

$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|    0     |   MoAI Accelerator  |  2xLarge.1024GB  |  -             |  -             |  -            |
|  * 1     |   MoAI Accelerator  |  xLarge.512GB    |  -             |  -             |  -            |
|    2     |   MoAI Accelerator  |  2xLarge.1024GB  |  -             |  -             |  -            |
|    3     |   MoAI Accelerator  |  8xLarge.4096GB  |  -             |  -             |  -            |
|    4     |   MoAI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

### Removing MoAI Accelerators: `moreh-smi device --rm`

This time, let's try to remove a specific accelerator corresponding to the specified device ID with the command `moreh-smi device --rm {Device_ID}`.

```json
$ moreh-smi --rm 1
+---------------------------------------------------+
|  Device  |        Name         |       Model      |
+===================================================+
|  * 0     |   MoAI Accelerator  |    Large.256GB   |
+---------------------------------------------------+
Remove device success.
```


The MoAI Accelerator with Device ID 1, [!badge variant="secondary" text=8xLarge.4096GB], has been removed using the above command. To confirm, when you run `moreh-smi` again, you will notice that the device has been removed.

### Other Various Options Utilization

`moreh-smi` provides various other options. You can use the `--help` option to see what options are available.

```
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

- `moreh-smi -p` - Monitor detailed hardware status of MoAI Accelerators.
- `moreh-smi -t` - Check MoAI Accelerator token information.

!!!info 
If you encounter issues during training, such as tangled processes or difficulty terminating, causing messages like "Process Running," use the `moreh-smi --reset` command.
!!!

------

## Changing MoAI Accelerators: `moreh-switch-model`

`moreh-switch-model` is a tool that allows you to change the flavor (specifications) of the currently configured MoAI Accelerator. By changing the flavor of the MoAI Accelerator, you determine how much GPU memory to use.

It can be used as follows:

For example, if the result of the `moreh-smi` command is as follows, it means that the "MoAI Platform AI Accelerator model currently set as the default is Accelerator 0, and this MoAI Accelerator is of type [!badge variant="secondary" text=Small.64GB] model."

```bash
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |   MoAI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
|    1     |   MoAI Accelerator  |  Medium.128GB    |  -             |  -             |  -            |
|    2     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     |   MoAI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

The `moreh-switch-model` command displays the following prompt:

```bash
$ moreh-switch-model
Current MoAI Accelerator: Medium.128GB

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

If you enter an integer corresponding to the model to be used from 1 to 13 (device number), the message "`The MoAI Platform AI Accelerator model is successfully switched to {model_id}."` will be displayed, and the MoAI Accelerator corresponding to the entered device number will be changed.


Let's change the MoAI Accelerator to [!badge variant="secondary" text=Large.256GB] as follows:

```bash
Selection (1-13, q, Q): 3
The MoAI Accelerator model is successfully switched to  "Large.256GB".

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

You can continue with the change or exit the MoAI Accelerator change by typing `q` or `Q`.

After the change is complete, when you run `moreh-smi` again to confirm, you will see the following result:

```bash
$ moreh-smi
+-----------------------------------------------------------------------------------------------------+
|                                                    Current Version: 24.5.0  Latest Version: 24.5.0  |
+-----------------------------------------------------------------------------------------------------+
|  Device  |        Name         |       Model      |  Memory Usage  |  Total Memory  |  Utilization  |
+=====================================================================================================+
|  * 0     |   MoAI Accelerator  |  Large.256GB     |  -             |  -             |  -            |
|    1     |   MoAI Accelerator  |  Medium.128GB    |  -             |  -             |  -            |
|    2     |   MoAI Accelerator  |  4xLarge.2048GB  |  -             |  -             |  -            |
|    3     |   MoAI Accelerator  |  Small.64GB      |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------------+
```

The MoAI Accelerator previously set as the `Small.64GB` model has been changed to the `Large.256GB` model.

------

## Updating MoAI Platform: `update-moreh`

`update-moreh` is a command that allows you to create a new conda environment and install MoAI Platform on it, or update the version of MoAI Platform already installed in the conda environment. You can use `update-moreh` in the following situations:

- If you have created a new conda environment and MoAI Platform Python packages need to be installed, you can easily install the latest version of MoAI Platform using the `update-moreh` command.
    
    ```bash
    $ conda create --name my_env python=3.8
    $ conda activate my_env
    $ update-moreh # Install MoAI Platform 
    
    Do you want to proceed? (y/n, default:n)
    y
    Moreh Framework installation start...
    Moreh Framework installation successful.
    ```
    
- If you want to use the latest version of MoAI Platform even in an existing conda environment where MoAI Platform is already installed, you can update the currently used MoAI Platform to the latest version using the `update-moreh` command alone.
    
    ```bash
    $ update-moreh # Update to the Latest Version
    Currently installed: 
    Possible upgrading version: 24.5.0
    
    Do you want to upgrade? (y/n, default:n)
    y
    Moreh Framework installation start...
    Moreh Framework installation successful.
    $ update-moreh
    Already installed : 24.5.0
    In some cases, you may need to install a specific version of MoAI Platform. In this case, you can specify the specific version of MoAI Platform you want to install using the 
    ```
    
- There may be cases where you need to install a specific version of the MoAI Platform. In such cases, you can use the **`--target`** option to specify the specific version you want to install.

```bash
update-moreh --target 24.5.0 # Install 24.5.0 version
```

- If the MoAI Platform is not functioning properly due to issues such as dependency conflicts between other packages in the conda environment, you may need to reconstruct the conda environment. In such cases, you can use **`update-moreh`** to restore the MoAI Platform within the conda environment. In the latter case, you can use the **`--force`** option to reconstruct the environment. (Can be used with the **`—-target`** option)

```bash
update-moreh --force --target 24.5.0
```
