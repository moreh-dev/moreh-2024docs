---
icon: terminal
tags: [ap]
order: 17
expanded: false
---

# AP 더 알아보기

AP와 관련된 로그를 조금 더 자세히 들여다보겠습니다.

```bash
[info] Got DBs from backend for auto config.
...
[info] The number of candidates is 30.
[info] Parallel Graph Compile start...
[info] Elapsed Time to compile all candidates = 6103 [ms]
[info] Parallel Graph Compile finished.
[info] The number of possible candidates is 7.
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 808 [ms]
[info] SelectBestGraphFromCandidates finished.
[info] Configuration for parallelism is selected.
[info] num_stages : 2, num_micro_batches : 4, batch_per_device : 1, No TP, recomputation : 0, distribute_param : true, distribute_low_prec_param : false
[info] train: true
```

MoAI Platform은 최적화된 병렬 처리를 위해 다양한 최적화 후보 config들을 생성합니다. 다음 로그는 Compiler Config Generator가 병렬화를 위한 후보 config를 30개로 설정했다는 의미입니다.

```bash
[info] The number of candidates is 30.
```

그 다음 모든 후보군에 대해서 연산 그래프를 생성합니다.

```bash
[info] Elapsed Time to compile all candidates = 6103 [ms]
```

위 로그를 통해 config 들을 compile 하는 데에 약 6.1초가 소요되었음을 알 수 있습니다.

여기서 다시 가능한 후보 config를 추정합니다.

```bash
[info] The number of possible candidates is 7.
```

이로써 총 7개의 가능한 config들이 있다는 것을 확인했습니다.

이제 graph simulator가 각 config에 대한 cost를 계산하고, 계산이 종료되면 최적의 config라고 판단한 1개의 config를 채택합니다.

```bash
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 808 [ms]
[info] SelectBestGraphFromCandidates finished.
```

위 로그는 최종 config 1개가 설정되기까지 cost를 계산하는데 약 0.8초가 소요되었음을 보여줍니다.

```bash
[info] Configuration for parallelism is selected.
[info] num_stages : 2, num_micro_batches : 4, batch_per_device : 1, No TP, recomputation : 0, distribute_param : true, distribute_low_prec_param : false
[info] train: true
```

이 정보는 `advanced_parallelization_selected_config.dump`라는 파일에 기록되며 파이썬 프로그램을 실행한 위치에 생성됩니다. 이제 `advanced_parallelization_selected_config.dump`이 어떻게 생겼는지 확인해 봅시다.

```bash
num_stages : 2, num_micro_batches : 4, batch_per_device : 1, No TP, recomputation : 0, distribute_param : true, distribute_low_prec_param : false
```

이처럼 **MoAI Platform에서는 단 한 줄의 프로그램 추가로 수 개의 병렬화 후보를 계산하여 최적의 병렬화 방법을 자동으로 선택할 수 있습니다.**

