---
icon: book
order: 2000
expanded: true
---

# MOREH DOCS

**MoAI(Moreh AI appliance for AI accelerators)** Platform is a scalable AI platform that allows for easy control of thousands of Graphics Processing Units (GPUs), which are required for developing large-scale deep learning models.

----

### Getting Started

   | 
---    | ---
 **[Get started with fine-tuning](Tutorials/index.md)** <br> MoAI Platform Beginner's Guide for Finetuning| [ **AP Guide**](/Supported_Documents/ap/index.md) <br> Advanced Parallelization (AP) Feature Instructions
[ **Moreh Toolkit Guide**](/Supported_Documents/moreh_toolkit.md) <br> Command Line Usage |[ **MoAI Platform Features**](/MoAI_Features/index.md) <br> The virtualization and parallelization features of the MoAI Platform

!!!primary Note
MoAI Platform is currently in ongoing development. Therefore, the contents of the document may change at any time.
!!!


## Core Technologies of MoAI Platform

![](./img/overview_01.png)

As deep learning models progress, they become increasingly complex and require significant computational resources as their parameters expand from billions to trillions. Developing large-scale models involves handling and processing an enormous volume of parameters, a task that is both daunting and time-consuming.

The MoAI Platform's automatic parallelization addresses these hurdles by simultaneously processing multiple tasks, determining the optimal computation methods for large models. This empowers AI engineers to concentrate solely on their core AI endeavors, regardless of the scale of their applications or the type of processors they use. Moreover, it efficiently utilizes GPU computational resources at a reasonable cost by allocating them only during calculation execution.

1. **[Support Various Accelerators with Single GPU Abstraction](https://docs.moreh.io/overview/#1-various-accelerators-multi-gpu-support)**
2. **[GPU Virtualization](https://docs.moreh.io/overview/#2-gpu-virtualization)**
3. **[Dynamic GPU Allocation](https://docs.moreh.io/overview/#3-dynamic-gpu-allocation)**
4. **[AI Compiler Automatic Parallelization](https://docs.moreh.io/overview/#4-ai-compiler-automatic-parallelization)**
---

