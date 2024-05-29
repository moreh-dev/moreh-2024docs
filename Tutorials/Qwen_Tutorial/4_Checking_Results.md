---
icon: terminal
tags: [tutorial, qwen]
order: 40
---

# 4. Checking Training Results

Running the **`train_qwen.py`** script, as in the previous chapter, will save the resulting model in the **`qwen_code_generation`** directory. This is a pure PyTorch model parameter file and is fully compatible with regular GPU servers, not just the MoAI Platform.

You can test the trained model using the **`inference_qwen.py`** script located under the **`tutorial`** directory in the GitHub repository you downloaded earlier. In the test, the prompt "Given a list of strings, create a function that joins them with spaces" was used.

```python
# tutorial/inference_qwen.py
...
input_text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a function to join given list of strings with space.

### Input:
['I', 'love', 'you']

### Output:
"""
```

Run the code below.

```bash
~/quickstart$ python tutorial/inference_qwen.py
```

Upon examining the output, you can confirm that the model has appropriately generated the function as per the prompt.

```
Qwen: Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a function to join given list of strings with space.

### Input:
['I', 'love', 'you']

### Output:
def join_strings(string_list):
    return ' '.join(string_list)

result = join_strings(['I', 'love', 'you'])
print(result)
```
