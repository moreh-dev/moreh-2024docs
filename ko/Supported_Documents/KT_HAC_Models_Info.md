---
icon: info
tags: [guide]
order: 30
---

# KT Hyperscale AI Computing (HAC) 서비스 가속기 모델 정보

- KT Cloud 공식 매뉴얼 바로가기
    - https://manual.cloud.kt.com/kt/hyperscale-ai-computing-howtouse-cj
- 현재 HAC 서비스는 AMD MI250 GPU를 사용해 구동되고 있습니다. 모델/애플리케이션에 따라 성능이 달라질 수 있지만 기본적으로 AMD MI250 하나와 NVIDIA A100 하나에서 동등한 성능이 나온다고 예상하시면 됩니다.

| Model | 실제 물리 GPU | 노드 수 |
| --- | --- | --- |
| Small.64GB | MI250 0.5개 | 1대 |
| Medium.128GB | MI250 1개 | 1대 |
| Large.256GB | MI250 2개 | 1대 |
| xLarge.512GB | MI250 4개 | 1대 |
| 2xLarge.1024GB | MI250 8개 | 2대 |
| 3xLarge.1536GB | MI250 12개 | 3대 |
| 4xLarge.2048GB | MI250 16개 | 4대 |
| 6xLarge.3072GB | MI250 24개 | 6대 |
| 8xLarge.4096GB | MI250 32개 | 8대 |
| 12xLarge.6144GB | MI250 48개 | 12대 |
| 24xLarge.12288GB | MI250 96개 | 24대 |
| 48xLarge.24576GB | MI250 192개 | 48대 |
| … | … | … |