---
icon: command-palette
tags: [guide]
order: 40
---


**LM(Large Model) 이란?**

*Moreh framework* 에서 학습, 추론이 가능한 대형 추론 모델을 의미합니다. 정기적으로 프레임워크와 함께 배포되며 Moreh 솔루션에서 딥러닝 학습에 필수적인 단계들을 수행할 수 있는 대형 언어 및 추론 모델을 다운로드할 수 있습니다. 따라서 사용자는 Large Model을 활용하여 직접 코딩하지 않아도 바로 학습, 추론을 수행할 수 있습니다.

Moreh 솔루션에서 지원하는 AI 프레임워크인 PyTorch와 TensorFlow, 그리고 학습 실행 방법을 설명 드리겠습니다.

## PyTorch

### 1. Large Language Model 코드 다운로드

아래 간단한 명령어 한 줄로 다양한 Large Model 코드를 얻게 됩니다.

```bash
# Resnet 모델 다운로드 예시
ubuntu@vm:~$ get-reference-model resnet
```

위와 같은 명령어 실행 시 ResNet에 대한 RM Code 설치 파일을 다운로드하고 실행하여 학습에 필요한 파일을 설치합니다. 명령어 실행 완료 시 모델명에 따른 폴더가 생성이 되며 해당 폴더로 들어가 아래 명령어로 바로 모델을 실행(학습)시킬 수 있습니다.

```bash
# 학습 모델 폴더로 이동
ubuntu@vm:~$ cd resnet

# 학습 모델 실행
ubuntu@vm:~$ python train.py
```

> get-reference-model 명령어는 sudo 없이 사용하시길 권장드립니다. sudo 명령어가 포함될 경우, 실행 시 아래와 같은 에러가 발생할 수 있습니다.
> 

```bash

    (...)

/data/work/install_arcface.sh: line 105: pip: command not found
- Installation Failed!

    (...)

```

### 2. Large Language Model 옵션 값 확인하기

현재 `get-reference-model` 에서 지원하는 옵션 값들을 보고 싶으시다면, 아무런 옵션 값을 주지 않고 실행하거나, `-h` 옵션을 주면 보실 수 있습니다.

```bash
ubuntu@vm:~$ get-reference-model -h
Usage: get-reference-model [-h|--help] [--download-only] [--download-dir] [-s|--show] (MODEL_NAME)
Example: get-reference-model resnet

Avaiable options:
-h, --help           Print help and exit
-s, --show           Print the available list of models
--download-only      Download the model shell script file without running
--download-dir       Set the installation path of a model default path: /home/ubuntu
--tensorflow         Set all option towards tensorflow reference model
                     ex) get-reference-model --tensorflow --show
                     ex) get-reference-model --tensorflow bert
```

### 3. 제공되는 모든 Large Model 목록 확인하기

현재 어떤 모델 코드들이 제공되는지 궁금하시다면 `—-show(또는 -s)` 옵션을 이용하여 확인할 수 있습니다. 가장 범용적으로 쓰이는 딥러닝 모델과 Moreh 솔루션을 이용한 딥러닝 학습 모범 사례로 쓰일만한 안전한 모델들이 목록에 나타납니다.

```bash
ubuntu@vm:~$ get-reference-model --show
[INFO] Downloadable Model List => 3dunet alexnet arcface bart bert dcgan deeplabv3m deeplabv3r densenet dlrm fasterrcnn fcn_resnet gnmt googlenet gpt gpt2 inceptionv3 lraspp maskrcnn mnasnet mobilenetv2 mobilenetv3 ncf resnet resnet2p1d resnet3d resnetMC resnext retinanet rnnt roberta shufflenetv2 speech2text squeezenet ssd ssdlite stdc t5 tacotron2 transformer transformerXL unet vgg wideresnet yolor yolov5
# 2023-09-11 기준 목록
```

### 4. Large Model 설치 파일 설정하기

모델 설치 파일 `(.sh)` 에 대해서 수정 사항이 필요할 경우엔 아래와 같이 `--download-only` 옵션을 추가하여 모델 설치 파일만 다운로드 하실수도 있습니다. 해당 옵션을 추가하고 실행하면 실행 경로에 `install_MODEL_NAME.sh` 파일이 생성됩니다.

다음은 `install_resnet.sh` 파일을 다운받는 명령어 예시입니다.

```bash
ubuntu@vm:~$ get-reference-model --download-only resnet
```

모델 설치 경로를 수정하고 싶으시다면 `—-download-dir` 옵션 값으로 모델 설치 경로를 수정하실 수 있습니다. 해당 옵션 값이 존재하지 않을 경우에는 기본 경로인 `/home/ubuntu` 에 설치가 됩니다.

> VM에서 /home의 기본 용량이 100GB이기 때문에 추가 디스크가 제공되는 경우가 있습니다. 위 설정을 통해 추가 디스크가 마운트된 디렉토리를 설정할 수 있습니다.
> 

다음은 HOME경로에 있는 `test` 폴더에 ResNet 모델을 설치하는 예시입니다.

```bash
ubuntu@vm:~$ get-reference-model resnet --download-dir ./test
```

`—-download-only` 옵션과 `—-download-dir` 옵션은 같이 사용하실 수 있습니다.

다음은 HOME경로에 있는 `test` 폴더에 `instsall_resnet.sh` 파일만 다운로드하는 명령어 예시입니다.

```bash
ubuntu@vm:~$ get-reference-model resnet --download-only --download-dir ./test
```

### 5. 모델 학습 시작하기

홈 디렉터리 아래의 해당 모델 디렉터리로 이동한 다음 train.py 스크립트를 실행하여 모델 학습을 시작할 수 있습니다.

```bash
(pytorch) ubuntu@vm:~$ cd ~/resnet
(pytorch) ubuntu@vm:~/resnet$ python train.py --train_batch_size 32

    (...)

[info] Requesting resources for KT AI Accelerator from the server...
[info] Initializing the worker daemon for KT AI Accelerator...
[info] [1/1] Connecting to resources on the server (192.168.00.00:00000)...
[info] Establishing links to the resources...
[info] KT AI Accelerator is ready to use.

    (...)

2023-09-12 16:59:04.998 | INFO     | __main__:train:355 - Epoch 1/42 start
2023-09-12 16:59:09.148 | INFO     | __main__:train:386 - TRAIN_STEP | Epoch:   1 | Iteration:    10/ 2012 | Loss:    4.702 | Throughput:  154.213 images/s | Duration: 4.150 s | Estimated Time Remaining: 830.850 s

    (...)

```

### Hyperparameter 변경하기

b 옵션은 mini-batch size, 즉 학습 이미지 몇 장을 한 번에 AI 가속기에서 학습시킬 것인지를 지정합니다. AI 가속기 사양이 높아질수록 거기에 맞춰 mini-batch size를 키워 주어야 최적의 성능을 얻을 수 있습니다. Hyperscale AI Computing의 AI 가속기 모델별로 권장하는 실행 옵션은 해당 [모델 매뉴얼](https://cloud.kt.com/solution/hyperscaleAiComputing/?tab=1)을 참고하십시오.

## Tensorflow

### 1. TensorFlow 가상 환경

처음 VM 생성 시 기본으로 `tensorflow` 이름의 Tensorflow용 conda 가상환경이 존재합니다. Tensorflow conda 환경이 없는 사용자 분들은 아래와 같은 방법으로 TensorFlow를 위한 가상환경을 생성하시기 바랍니다.

### 2. Tensorflow Large Model 코드 다운로드

`get-reference-model` 명령어 한 줄로 다양한 Large Model(이하 LM) Code를 얻게 됩니다.

```bash
# Resnet 모델 다운로드 예시
ubuntu@vm:~$ get-reference-model --tensorflow resnet
```

위와 같은 명령어 실행 시 ResNet에 대한 Large Model Code 설치 파일 및 샘플 데이터을 다운로드하게 되며, 동시에 해당 설치 파일을 실행시켜 실행환경을 세팅해줍니다. 명령어 실행 완료 시 모델명에 따른 폴더가 생성이 되며 해당 폴더로 들어가 아래 명령어로 바로 모델을 실행(학습)시킬 수 있습니다.

```bash
# 학습 모델 폴더로 이동
ubuntu@vm:~$ cd resnet

# 학습 모델 실행
ubuntu@vm:~$ python train.py
```

### 3. Large Model 옵션 값 확인하기

현재 `get-reference-model`에서 지원하는 옵션 값들을 보고 싶으시다면, 아무런 옵션 값을 주지 않고 실행하거나, `-h`옵션을 주면 보실 수 있습니다.

```bash
ubuntu@vm:~$ get-reference-model -h

Usage: get-reference-model [-h|--help] [--download-only] [--download-dir] [-s|--show] (MODEL_NAME)
Example: get-reference-model resnet

Avaiable options:
-h, --help           Print help and exit
-s, --show           Print the available list of models
--download-only      Download the model shell script file without running
--download-dir       Set the installation path of a model default path: /home/ubuntu
--tensorflow         Set all option towards tensorflow reference model
                    ex) get-reference-model --tensorflow --show
                    ex) get-reference-model --tensorflow bert
```

### 4. Large Model 설치 파일 설정하기

모델 설치 경로를 수정하고 싶으시다면 `—-download-dir` 옵션 값으로 모델 설치 경로를 수정하실 수 있습니다. 해당 옵션 값이 존재하지 않을 경우에는 기본 경로인 `/home/ubuntu`에 설치가 됩니다.

> VM에서 /home의 기본 용량이 100GB이기 때문에 추가 디스크가 제공되는 경우가 있습니다. 위 설정을 통해 추가 디스크가 마운트된 디렉토리를 설정할 수 있습니다.
> 

다음은 `/data/tf-rm` 경로에 ResNet 모델 파일을 다운받는 명령어 예시입니다.

```bash
ubuntu@vm:~$ get-reference-model --tensorflow resnet --download-dir /data/tf-rm
```

### 5. 모델 학습 시작하기

홈 디렉터리 아래의 해당 모델 디렉터리로 이동한 다음 train.py 스크립트를 실행하여 모델 학습을 시작할 수 있습니다.

```bash
(tensorflow) ubuntu@vm:~$ cd ~/resnet
(tensorflow) ubuntu@vm:~/resnet$ python train.py --train_batch_size 32

    (...)

[info] Requesting resources for KT AI Accelerator from the server...
[info] Initializing the worker daemon for KT AI Accelerator...
[info] [1/1] Connecting to resources on the server (192.168.00.00:00000)...
[info] Establishing links to the resources...
[info] KT AI Accelerator is ready to use.

    (...)

| INFO | moreh_controller.py:_train_n_steps:474 TRAIN_STEP | Iteration : 100/69300 | Loss : 5.113 | Throughput : 462.375 samples/s | Duration : 83.049 s | Estimated Time Remaining : 22063.482 s

    (...)

```

### Hyperparameter 변경하기

b 옵션은 mini-batch size, 즉 학습 이미지 몇 장을 한 번에 AI 가속기에서 학습시킬 것인지를 지정합니다. AI 가속기 사양이 높아질수록 거기에 맞춰 mini-batch size를 키워 주어야 최적의 성능을 얻을 수 있습니다. Hyperscale AI Computing의 AI 가속기 모델별로 권장하는 실행 옵션은 해당 [모델 매뉴얼](https://cloud.kt.com/solution/hyperscaleAiComputing/?tab=1)을 참고하십시오.