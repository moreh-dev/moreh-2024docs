---
order: 90
icon: quote
tags: [guide]
---


# Text Generate AI

### Chatbot (Small, Medium, Large)

하단 Textbox에 프롬프트를 입력하면 MAMH가 학습한 정보를 사용하여 답변을 제공합니다.

다음은 한글/영어 특화 언어 모델을 사용한 Chatbot_Small, Chatbot_Medium, Chatbot_Large에서 시도해 볼 수 있는 몇 가지 **프롬프트 예시**입니다.

(한글 프롬프트 예시)

- “닭이 먼저야, 달걀이 먼저야?”
- “특정 사회 문제를 다루는 비영리 단체를 위한 홍보 캠페인을 계획해줘.”
- “단편 소설의 제목을 브레인스토밍하는 데 도움을 줘.”

(영문 프롬프트 예시)

- Tell me the best places to travel on a budget of $1000 including car hire, flights and accommodation.
- Write a summary of “The catcher in the rye”
- Write me a step-by-step guide to use “AI Model Hub”
- Generate a list of at least 10 keywords related to Mother nature

### Code Generator

다음은 Code Generator에서 시도해 볼 수 있는 프롬프트 예시입니다.

(영문 프롬프트 예시)

- Train a logistic regression model, predict the labels on the test set and compute the accuracy score
    
    ```python
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1)
    ```
    
- Continue writing this code in Python
    
    ```python
    #function that determines if a year is a leap year or not
    def is_leap_year(year):
        if year % 4 == 0:
            if year % 100 == 0:
    ```
    
- Generate a function ***to find the value for (a+b)^3 using lambda.***
- Generate a Docker script to create a linux machine that has python 3.11.7
installed with following libraries: Pytorch, Tensorflow, Scikit- learn, pandas, numpy.
- Generate a unit test for a function that determines if a year is a leap year or not.

(한글 프롬프트 예시)

- lambda를 사용하여 (a+b)^3의 값을 찾는 함수를 작성해봐.
- 배열을 인수로 받아 가장 큰 값을 반환하는 함수(함수 이름: findMax) 를 작성해줘.
- 파이썬 3.11.7이 설치된 Linux 머신을 생성하기 위한 도커 스크립트를 짜줘.