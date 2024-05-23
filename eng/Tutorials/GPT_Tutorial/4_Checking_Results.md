---
icon: terminal
tags:  [tutorial, gpt]
order: 40
---

# 4. Checking Training Results

As in the previous chapter, when you run the `train_gpt.py` script, the resulting model will be saved in the `code_generation` directory. This is a pure PyTorch model parameter file and is fully compatible compatible not only with MoAI Platform but also with regular GPU servers.

You can test the trained model using the **`inference_gpt.py`** script located under the **`tutorial`** directory of the GitHub repository you downloaded beforehand.

```python
# inference_gpt.py
...
QUERY = """Write a python program that counts all 'a's in a string. For example, if the string "Banana" is given, the program should return 3.
"""
```

Run the train script.

```bash
~/quickstart$ python tutorial/inference_gpt.py
```

Upon inspecting the output, you can confirm that the model has generated an appropriate function based on the prompt content.

```python

def count_a(string):
    count = 0
    for char in string:
        if char == 'a':
            count += 1
    return count

string = "Banana"
print(count_a(string))</s>

Output:
3</s>

Explanation:
The program defines a function called count_a that takes a string as an argument. It initializes a count variable to 0, which will be used to keep track of the number of 'a's found in the string. Then, it iterates through each character in the string using a for loop. If the character is equal to 'a', the count is incremented by 1. Finally, the function returns the count.

In the given example, the string "Banana" is passed to the count_a function, and the program prints 3, which is the correct output.</s>

Note: The program assumes that there will always be at least one 'a' in the string. If you want to handle the case when the string is empty, you can add a check at the beginning of the function and return 0 in that case.</s>
```

