---
icon: terminal
tags: [guide]
order: 18
expanded: false
visibility: hidden
---

# Advanced Settings for AP

While simply adding **`torch.moreh.option.enable_advanced_parallelization()`** allows you to use the basic AP functionality, you can easily customize the parallelization feature according to your preferences using various variables provided by the MoAI Platform.

## Customizing AP Configuration

When using the AP feature as an API in a Python program, you can set additional arguments to restrict specific configurations.

```python
def main(args):

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization( 
				num_stages=2,
				num_micro_batches=32,
				activation_recomputation=True,
				distribute_parameter=True,
		)
```

Below are the configurable variables that can be inputted into the API, allowing users to optimize distributed parallelization according to their needs.

- **`pipeline_parallel`** (*bool*, Default: *true*): Whether Pipeline Parallel is applied
- **`num_stages`** (*str, int*,*** default: *‘auto’*): Maximum number of stages in Pipeline Parallelism.
- **`num_micro_batches`**(*str, int*, Default: *‘auto’*):  Number of micro-batches in Pipeline Parallelism.
- **`activation_recomputation`** (*str*, *bool*, Default: *‘auto’*): Whether activation recomputation is applied
- **`distribute_parameter`**(*str*, *bool*, Default: *‘auto’*): Whether the feature of distributing param and grad to GPU is applied
- **`mixed_precision`** (*bool*, Default: *true*): Whether bfloat16 is applied

## **Environment Variables for Performance and Log Information of AP**

AP generates multiple candidate configurations and calculates costs based on them. The speed of this process and the available configurations may vary depending on the hardware resources used by the user.

- **`MOREH_ADVANCED_PARALLELIZATION_MAX_PARALLEL_COMPILE_THREADS`**
    - value type = int
    - default = 16
    - Specifies the number of threads used by the compiler during compilation.
    - If the waiting time during compilation is long, it is recommended to increase this value and retry.
        - However, compile time may vary depending on the CPU usage and number of CPU cores.
        - Therefore, increasing this value may not necessarily improve compilation speed.
- **`MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG`**
    - default = 0
    - Provides additional information during Advanced Parallelization compilation.
        - If **`MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG=1`**, it will be printed to the console.
        - If **`MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG=2`**, it will be saved in the autoconfig_log.dump format.
- **`MOREH_ADVANCED_PARALLELIZATION_MEMORY_USAGE_CORRECTION_RATIO`**
    - default = 80
    - Represents the available memory of the GPU used during compilation in Advanced Parallelization.
    - For example, the default setting limits the available memory to 80% of the actual GPU memory.

These environment variables can be configured as follows.

```bash
$ export MOREH_ADVANCED_PARALLELIZATION_MAX_PARALLEL_COMPILE_THREADS=16
$ export MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG=1
$ export MOREH_ADVANCED_PARALLELIZATION_MEMORY_USAGE_CORRECTION_RATIO=80
```
