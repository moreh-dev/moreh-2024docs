---
icon: container
tags: [guide]
order: 50
---


## K8S Cluster에 접근하기 위한 서버 접속

- K8S Cluster를 사용하기 위해 `moreh-k8s-master-vm01` 서버로 접속해야합니다.
- 관리자가 이용자에게 제공한 VM 접속정보(IP, Port, SSH Key)를 입력해 접속할 수 있습니다.
- 관리자가 해당 VM 접속정보를 이용자에게 제공하였다고 가정해보겠습니다.
    
    ```yaml
    hostname: moreh-k8s-master-vm01
    ip: xxx.00.xx.000
    port: 22
    user: ubuntu
    permission key: key.pem
    ```
    
- 다음은 VM 접속정보를 활용하여 VM에 ssh 명령어로 접속하는 예시입니다.
    
    ```bash
    (pytorch) $ ssh -i key.pem -p 22 ubuntu@xxx.00.xx.000
    (pytorch) ubuntu@moreh-k8s-master-vm01:~$
    ```
    

## Pod을 띄우기 위한 Manifest 파일 작성

- Manifest 파일은 쿠버네티스 오브젝트를 관리하기 위한 선언적 specification을 포함하는 YAML 파일입니다.
- K8S 클러스터 안에서 pod을 띄우기 위해 다음과 같이 작성합니다.
    
    ```yaml
    # sample.yaml
    apiVersion: apps/v1
    
    # Pod의 선언적 업데이트와 스케일링을 관리하는 'Deployment' manifest
    kind: Deployment
    
    # 오브젝트 정의
    metadata:
    name: {pod 이름}
    namespace: maf-service
    labels:
        app: moreh
    spec:
    # pod 인스턴스 갯수
    replicas: 1
    selector:
        matchLabels:
        app: moreh
    template:
        metadata:
        labels:
            app: moreh
        spec:
        containers:
        - name: {컨테이너 이름}
            imagePullPolicy: IfNotPresent
            image: {Moreh 솔루션의 Docker Image}
            command: ["/bin/sh", "-c", "echo 'eth0' > /etc/moreh/net_interfaces && tail -f /dev/null"]
            env:
            - name: MOREH_SDA_TOKEN
            value: {SDA 토큰}
    ```
    
- 사용자가 작성해야 할 주요 key는 다음과 같습니다.
    - `metadata.name` : pod의 이름
    - `spec.template.spec.containers[0].name` : container의 이름
    - `spec.template.spec.containers[0].image` : Moreh 솔루션의 도커 이미지
    - `spec.template.spec.containers[0].env[0].value` : 사용할 토큰
- K8S 클러스터에 pod을 생성하기 위해 다음의 명령어로 위에서 작성한 manifest 파일을 적용합니다.
    - `$ kubectl apply -f {파일 경로}`
    
    ```bash
    (pytorch) ubuntu@moreh-k8s-master-vm01:~$ kubectl apply -f sample.yamldeployment.apps/sample-pod created
    ```
    
    - `apply` : 해당 액션으로 쿠버네티스 리소스를 생성/수정
    - `f` : 파일 경로
    - K8S 클러스터에 해당 pod이 생성되었는지 다음의 명령어로 확인합니다.
    - `$ kubectl get pods -n {metadata.namespace}`
    
    ```bash
    (pytorch) ubuntu@moreh-k8s-master-vm01:~$ kubectl get pods -n maf-serviceNAME                          READY   STATUS    RESTARTS   AGEsample-pod-866dc56b48-th68r   1/1     Running   0          21s
    ```
    
- 생성한 pod에 다음의 명령어로 접속합니다.
    - `$ kubectl exec -it {pod 이름} -n {namespace} -c {container 이름} -- /bin/bash`
    
    ```bash
    (pytorch) ubuntu@moreh-k8s-master-vm01:~/$ kubectl exec -it sample-pod-866dc56b48-th68r -n maf-service -c sample-container -- /bin/bash
    (moreh) root@sample-pod-866dc56b48-th68r:~#
    ```
    
    ---
    
    ## Pod 안에서 모레 솔루션 활용
    
- 먼저 모레 솔루션이 작동하는지 여부를 moreh toolkit을 활용해 확인해봅니다.

### Moreh Toolkit 사용

- `moreh-smi`

```bash
(moreh) root@sample-pod-866dc56b48-th68r:~# moreh-smi15:51:50 September 10, 2023+-------------------------------------------------------------------------------------------------+|                                                Current Version: 23.9.0  Latest Version: 23.9.0  |+-------------------------------------------------------------------------------------------------+|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |+=================================================================================================+|    0     |  KT AI Accelerator  |  Small.64GB  |  -             |  -             |  -            |+-------------------------------------------------------------------------------------------------+
```

- `moreh-switch-model`

```bash
(moreh) root@sample-pod-866dc56b48-th68r:~# moreh-switch-modelCurrent KT AI Accelerator: Small.64GB1. Small.64GB  *2. Medium.128GB3. Large.256GB4. xLarge.512GB5. 1.5xLarge.768GB6. 2xLarge.1024GB7. 3xLarge.1536GB8. 4xLarge.2048GB9. 6xLarge.3072GB10. 8xLarge.4096GB11. 12xLarge.6144GB12. 24xLarge.12288GB13. 48xLarge.24576GBSelection (1-13, q, Q):
```

- 만약 moreh tools를 실행했을 때 다음과 같이 ‘moreh::InvalidToken’ 에러가 발생한 경우 토큰 설정을 해주어야 합니다.
    
    ```bash
    (moreh) root@sample-pod-866dc56b48-th68r:~# moreh-switch-modelterminate called after throwing an instance of 'moreh::InvalidToken'what():  The SDA token is not set.Aborted (core dumped)
    ```
    
    - 컨테이너 내부에서 `/etc/moreh/token`에 할당받은 토큰을 입력하면 위의 문제가 해결됩니다.
    
    ```bash
    (moreh) root@sample-pod-866dc56b48-th68r:/etc/moreh# vi token------------------------------------------------------THIS_IS_YOUR_TOKEN~~~
    ```
    

### Duplicable SDA 설정

- 추론 시스템을 구축하는 경우, 하나의 SDA에서 여러 프로세스를 만들 필요가 있을 수 있습니다. 이러한 경우 duplicable SDA 설정을 통해 하나의 SDA에서 다수의 GPU 활용 프로그램을 실행할 수 있습니다.
- Duplicable SDA는 `moreh-smclient`를 통해 설정할 수 있습니다.

### PyTorch 학습

- 샘플 코드인 pytorch-sample.py를 사용해 학습을 진행하면 다음과 같은 결과를 얻을 수 있습니다.

```bash
(moreh) root@sample-pod-866dc56b48-th68r:~# python pytorch-sample.py
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
100.0%Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
111.0%Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
100.0%Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
159.1%Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
Processing...
Done!
[2023-09-10 15:55:21.504] [info] Requesting resources for KT AI Accelerator from the server...
[2023-09-10 15:55:22.521] [warning] A newer version of Moreh AI Framework is available. You can update the software to the latest version by running "update-moreh".
[2023-09-10 15:55:22.521] [info] Initializing the worker daemon for KT AI Accelerator
[2023-09-10 15:55:23.924] [info] [1/1] Connecting to resources on the server (192.168.110.21:24174)...
[2023-09-10 15:55:23.928] [info] Establishing links to the resources...
[2023-09-10 15:55:24.092] [info] KT AI Accelerator is ready to use.
Epoch 1
-------------------------------
loss: 2.295182  [    0/60000]
loss: 2.294096  [ 6400/60000]
loss: 2.284728  [12800/60000]
loss: 2.287824  [19200/60000]
loss: 2.250205  [25600/60000]
loss: 2.257666  [32000/60000]
loss: 2.233773  [38400/60000]
loss: 2.224680  [44800/60000]
loss: 2.241890  [51200/60000]
loss: 2.213466  [57600/60000]
Test Error:
 Accuracy: 47.3%, Avg loss: 0.034640

Epoch 2
-------------------------------
loss: 2.217214  [    0/60000]
loss: 2.211718  [ 6400/60000]
loss: 2.190849  [12800/60000]
loss: 2.195787  [19200/60000]
loss: 2.104573  [25600/60000]
loss: 2.157143  [32000/60000]
loss: 2.090617  [38400/60000]
loss: 2.084928  [44800/60000]
loss: 2.137709  [51200/60000]
loss: 2.062753  [57600/60000]
Test Error:
 Accuracy: 49.8%, Avg loss: 0.032298
```

---

## 사용 완료한 Pod 제거

- pod을 제거하기 위해서는 먼저 deployment를 삭제해야합니다.
    - `$ kubectl delete pod {pod 이름} -n {namespace}` 로 pod을 삭제하면 deployment 컨트롤러가 새 pod을 생성하여 복제본 수를 유지하려고 하기 때문입니다.
- 다음의 명령어로 deployment를 삭제합니다.
    - `$ kubectl delete deployment {deployment 이름} -n {namespace}`
    
    ```bash
    (pytorch) ubuntu@moreh-k8s-master-vm01:~$ kubectl delete deployment sample-pod -n maf-servicedeployment.apps "sample-pod" deleted
    ```
    
- 그 후 pod을 제거합니다.
    - `$ kubectl delete pod {pod 이름} -n {namespace}`
    
    ```bash
    (pytorch) ubuntu@moreh-k8s-master-vm01:~$ kubectl delete pod sample-pod-866dc56b48-kslxt -n maf-servicepod "sample-pod-866dc56b48-kslxt" deleted
    ```
    
- pod이 삭제되었는지 확인합니다.
    - `$ kubectl get pods -n {namespace}`
    
    ```bash
    (pytorch) ubuntu@moreh-k8s-master-vm01:~$ kubectl get pods -n maf-serviceNAME                      READY   STATUS    RESTARTS   AGE
    ```