---
icon: terminal
tags: [ap]
order: 18
expanded: false
visibility: private
---

# AP 기능 고급 설정

`torch.moreh.option.enable_advanced_parallelization()` 한 줄을 추가하는 것만으로도 기본적인 AP 기능을 사용하실 수 있지만, MoAI Platform에서 제공하는 다양한 변수를 활용해 사용자가 원하는 방식으로 손쉽게 병렬화 기능을 이용하실 수 있습니다.

## AP의 config 커스터마이징

AP 기능을 다음과 같이 python 프로그램에서 API로 사용할 때, 특정 config를 제한하도록 추가 인자를 설정할 수 있습니다.

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

아래는 API에 입력할 수 있는 config 변수들입니다.  다음 인자들을 활용해 사용자가 원하는 방식으로 분산 병렬화를 최적화할 수 있습니다. 

- **`pipeline_parallel`** (*bool*, Default: *true*) - Pipeline Parallel 사용 여부
- **`num_stages`** (*str, int*,*** default: *‘auto’*) - Pipeline Parallel에서 최대 stage 수
- **`num_micro_batches`**(*str, int*, Default: *‘auto’*): pipeline parallel의 micro batch 수
- **`activation_recomputation`** (*str*, *bool*, Default: *‘auto’*) activation recomputation 사용 여부
- **`distribute_parameter`**(*str*, *bool*, Default: *‘auto’*): param, grad를 GPU 분배하는 기능 사용 여부
- **`mixed_precision`** (*bool*, Default: *true*) - bfloat16 사용 여부

## AP의 성능 및 로그 정보를 변경할 수 있는 환경 변수

AP는 여러 후보 config들을 생성하고, 이를 통해 cost를 계산하는 작업을 거칩니다. 이 과정은 사용자가 사용하는  하드웨어 리소스에 따라 그 실행 속도 및 가능한 config가 달라집니다.

- **`MOREH_ADVANCED_PARALLELIZATION_MAX_PARALLEL_COMPILE_THREADS`**
    - value type = int
    - default = 16
    - Compiler가 compile 시 사용하는 thread 의 개수 입니다.
    - Compile 시 대기 시간이 길어질 경우 해당 값을 높여 재실행하시길 권장드립니다.
        - 다만 사용 중인 CPU 사용량, CPU core 개수에 따라 compile 시간이 변동될 수 있습니다.
        - 따라서 해당 수치를 올려도 compile 속도가 향상되지 않을 수 있습니다.
- **`MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG`**
    - default = 0
    - Advanced Parallelization 시 compile 되는 추가 정보를 제공합니다.
        - `MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG=1`일 경우 console 에 출력합니다.
        - `MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG=2` 일 경우 autoconfig_log.dump 형태로 저장됩니다.
- **`MOREH_ADVANCED_PARALLELIZATION_MEMORY_USAGE_CORRECTION_RATIO`**
    - default = 80
    - Advance Parallelization 에서 compile 할때 사용하는 GPU의 가용 메모리양입니다.
    - 예를 들어, 기본 설정에서 가용 메모리양은 실제 GPU 메모리의 80%로 제한합니다.

위 환경 변수들은 터미널에서 다음과 같은 방법으로 설정하실 수 있습니다.

```bash
$ export MOREH_ADVANCED_PARALLELIZATION_MAX_PARALLEL_COMPILE_THREADS=16
$ export MOREH_ADVANCED_PARALLELIZATION_DETAILED_LOG=1
$ export MOREH_ADVANCED_PARALLELIZATION_MEMORY_USAGE_CORRECTION_RATIO=80
```

