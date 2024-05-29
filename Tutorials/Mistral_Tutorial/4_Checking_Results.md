---
icon: terminal
tags: [tutorial, mistral]
order: 40
---

# 4. Checking Training Results

Running the **`train_mistral.py`** script, as in the previous section, will save the resulting model in the **`mistral_code_generation`** directory. This is a pure PyTorch model parameter file and is fully compatible with regular GPU servers, not just the MoAI Platform.

You can test the trained model using the **`inference_mistral.py`** script located in the **`tutorial`** directory of the GitHub repository you downloaded earlier. In this test, the prompt "Create a function that takes a list of strings as input and joins them with spaces" was used.

```python
# tutorial/inference_mistral.py
...
input_text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a function to join given list of strings with space.

### Input:\n['I', 'love', 'you']

### Output:
"""
```

Run the code below.

```python
~/quickstart$ python tutorial/inference_mistral.py
```

Upon examining the output, you can confirm that the model has appropriately generated the function as per the prompt.

```
Mistral: Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a function to join given list of strings with space.

### Input:
['I', 'love', 'Moreh']

### Output:
def join_strings(string_list):
		return ' '.join(string_list)

result = join_strings(['I', 'love', 'Moreh'])
print(result)
```
