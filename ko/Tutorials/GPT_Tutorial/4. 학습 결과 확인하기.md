---
icon: terminal
tags: [guide]
order: 40
---

# 4. 학습 결과 확인하기

앞 장과 같이 `train_gpt.py` 스크립트를 실행하면 결과 모델이 `code_generation` 디렉토리에 저장됩니다. 이는 순수한 PyTorch 모델 파라미터 파일로 MoAI Platform이 아닌 일반 GPU 서버에서도 100% 호환됩니다.

미리 다운로드한 GitHub 레포지토리의 `tutorial` 디렉토리 아래에 있는 `inference_gpt.py`  스크립트로 학습된 모델을 테스트해 볼 수 있습니다.

```python
# inference_gpt.py
...
QUERY = """Write a python program that counts all 'a's in a string. For example, if the string "Banana" is given, the program should return 3.
"""
```

코드를 실행합니다.

```bash
~/quickstart$ python tutorial/inference_gpt.py
```

출력값을 확인해보면 모델이 프롬프트 내용대로 적절한 함수를 생성한 것을 확인할 수 있습니다.

```bash

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
