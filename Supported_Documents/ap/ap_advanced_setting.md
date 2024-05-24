## Advanced Settings for AP

Adding just one line of ``torch.moreh.option.enable_advanced_parallelization()`` enables basic AP functionality, but with MoAI Platform, users can easily leverage parallelization features in the way they want using various variables provided.

Customizing AP Configuration
When using the AP feature as an API in a Python program, you can set additional arguments to restrict specific configurations.

```python
Copy code
def main(args):

    # Apply Advanced Parallelization
    torch.moreh.option.enable_advanced_parallelization( 
				num_stages=2,
				num_micro_batches=32,
				activation_recomputation=True,
				distribute_parameter=True,
		)
```
Below are the config variables that can be input into the API. With these arguments, users can optimize distributed parallelization according to their preferences.

아래는 API에 입력할 수 있는 config 변수들입니다.  다음 인자들을 활용해 사용자가 원하는 방식으로 분산 병렬화를 최적화할 수 있습니다. 

- **`pipeline_parallel`** (*bool*, Default: *true*) - Pipeline Parallel([Gpipe](https://blog.research.google/2019/03/introducing-gpipe-open-source-library.html)) 사용 여부
- **`num_stages`** (*str, int*,*** default: *‘auto’*) - Pipeline Parallel에서 최대 stage 수
- **`num_micro_batches`**(*str, int*, Default: *‘auto’*): pipeline parallel의 micro batch 수
- **`activation_recomputation`** (*str*, *bool*, Default: *‘auto’*) activation recomputation 사용 여부
- **`distribute_parameter`**(*str*, *bool*, Default: *‘auto’*): param, grad를 GPU 분배하는 기능 사용 여부
- **`mixed_precision`** (*bool*, Default: *true*) - bfloat16 사용 여부

## AP의 성능 및 로그 정보를 변경할 수 있는 환경 변수

Environment Variables for Changing AP Performance and Log Information
AP generates multiple candidate configurations and calculates costs based on them. This process and the available configurations depend on the hardware resources used by the user, leading to varying execution speeds and available configurations.

MOREH_ADVANCED_PARALLELIZATION_MAX_PARALLEL_COMPILE_THREADS
value type = int
default = 16
Number of threads used by the compiler during compilation.
It is recommended to increase this value if compilation time is long.
However, compilation time may vary depending on CPU usage and number of CPU cores.
Therefore, increasing this value may not necessarily improve compilation speed.
MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG
default = 0
Provides additional information during Advanced Parallelization compilation.
If MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG=1, it will be output to the console.
If MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG=2, it will be saved in the form of autoconfig_log.dump.
MOREH_ADVANCED_PARALLELIZATION_MEMORY_USAGE_CORRECTION_RATIO
default = 80
The amount of GPU memory used during compilation in Advanced Parallelization.
For example, the default setting limits the available memory to 80% of the actual GPU memory.
These environment variables can be set in the terminal as follows.