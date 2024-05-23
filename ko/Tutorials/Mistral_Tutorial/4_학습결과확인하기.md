---
icon: terminal
tags: [guide]
order: 40
---

# 4. 학습 결과 확인하기

앞 장과 같이 `train_mistral.py` 스크립트를 실행하면 결과 모델이 `mistral_code_generation` 디렉터리에 저장됩니다. 이는 순수한 PyTorch 모델 파라미터 파일로 MoAI Platform이 아닌 일반 GPU 서버에서도 100% 호환됩니다.

미리 다운로드한 GitHub 레포지토리의 `tutorial` 디렉토리 아래에 있는 `inference_mistral.py` 스크립트로 학습된 모델을 테스트해 볼 수 있습니다. 테스트에는 ‘주어진 문자열 리스트를 입력 받아 공백으로 결합하는 함수를 만들어’라는 프롬프트가 사용되었습니다.

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

코드를 실행합니다.

```python
~/quickstart$ python tutorial/inference_mistral.py
```

출력값을 확인해보면 모델이 프롬프트 내용대로 적절한 함수를 생성한 것을 확인할 수 있습니다.

```bash
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
