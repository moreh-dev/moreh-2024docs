---
icon: terminal
tags: [guide]
order: 17
expanded: false
---

# Learn More About AP

Let's take a closer look at the logs related to AP.

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

The MoAI Platform generates various optimized configurations for parallel processing to find the best optimization. The following log indicates that the Compiler Config Generator has set the number of candidate configurations for parallelization to 30.

```bash
[info] The number of candidates is 30.
```

Then, it generates an operation graph for each candidate.

```bash
[info] Elapsed Time to compile all candidates = 6103 [ms]
```

From the above log, we can see that it took approximately 6.1 seconds to compile the configurations.

Next, it estimates the possible candidate configurations again.

```bash
[info] The number of possible candidates is 7.
```

Thus, it confirms that there are a total of 7 possible configurations.

Now, the graph simulator calculates the cost for each configuration, and once the calculation is complete, it selects the optimal configuration as the final choice.

```bash
[info] SelectBestGraphFromCandidates start...
[info] Elapsed Time to compute cost for survived candidates = 808 [ms]
[info] SelectBestGraphFromCandidates finished.
```

The log above shows that it took about 0.8 seconds to calculate the cost until one final configuration was selected.

```bash
[info] Configuration for parallelism is selected.
[info] num_stages : 2, num_micro_batches : 4, batch_per_device : 1, No TP, recomputation : 0, distribute_param : true, distribute_low_prec_param : false
[info] train: true
```

This information is recorded in a file named **`advanced_parallelization_selected_config.dump`**, which is created in the location where the Python program is executed. Now, let's see how **`advanced_parallelization_selected_config.dump`** looks like.

```bash
num_stages : 2, num_micro_batches : 4, batch_per_device : 1, No TP, recomputation : 0, distribute_param : true, distribute_low_prec_param : false
```

In this way, by adding just one line of code, it is possible to compute multiple parallelization candidates and achieve optimal parallelization.
