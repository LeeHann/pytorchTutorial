"""
[모델 매개변수 최적화하기]
데이터 매개변수를 최적화해 모델을 학습하고 검증, 테스트해보자.

모델은 매 학습 반복 과정(epoch, 1회 학습 주기)에서 정답을 추측하고, 정답과 예측값 사이 오류(loss, 손실)을 계산하고,
매개변수에 대한 손실함수의 도함수를 수집해 경사 하강법(SGD, Stochastic Gradient Descent)으로 매개변수를 최적화한다.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

"""
하이퍼파라미터(Hyper parameter)

모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수이다.
서로 다른 하이퍼파라미터 값은 모델 학습과 수렴율에 영향을 미칠 수 있다.
하이퍼 파라미터의 종류는 아래와 같다.
1. 에폭(epoch)수 : 데이터셋을 반복하는 횟수
2. 배치 크기(batch size) : 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
3. 학습률(learning rate) : 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 
                값이 작을수록 학습 속도가 느려지고, 값이 크면 예측할 수 없는 동작이 발생할 수 있다.
"""
# ex.
learning_rate = 1e-3
batch_size = 64
epochs = 5

"""
최적화 단계(Optimization Loop)

하이퍼파라미터를 설정하고 최적화단계를 통해 모델을 학습하고 최적화할 수 있다.
이때 최적화 단계의 각 반복을 에폭이라 하는데, 
에폭은 학습(train loop, 학습용 데이터셋을 반복하고 최적의 매개변수로 수렴한다.),
검증/테스트(validation/test loop, 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복한다.) 단계로 구성된다.

학습 단계에서는 손실함수(loss function)을 통해 신경망이 추측한 정답과 정답레이블의 손실(loss)를 계산한다.
일반적인 손실함수에는 회귀 문제(regression task)에 사용하는 nn.MSELoss(평균 제곱 오차(MSE; Mean Square Error))나
분류 문제에 사용하는 nn.NLLLoss(음의 로그 우도(Negative Log Likelihood), 
그리고 nn.LogSoftmax와 nn.NLLLoss를 합친 nn.CrossEntropyLoss 등이 있다.

모델의 출력 로짓(logit)을 nn.CrossEntropyLoss에 전달해 로짓을 정규화하고 예측 오류를 계산한다.

로짓(logit)은 log + odds에서 나온 말. 
오즈는 그 값이 1보다 큰지가 결정의 기준이고, 로짓은 0보다 큰지가 결정의 기준. 
확률의 범위는 [0,1] 이나, 로짓의 범위는 [−∞,∞] 이다.
[https://haje01.github.io/2019/11/19/logit.html]
"""

# 손실 함수를 초기화합니다.
loss_fn = nn.CrossEntropyLoss()

"""
최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정이다.
최적화 알고리즘은 이 매개변수를 조정하는 과정을 정의한다. (여기서는 확률적 경사하강법(SGD; Stochastic Gradient Descent))
모든 최적화 절차는 optimizer 객체에 캡슐화된다.
"""

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""
학습 단계에서 최적화는 3단계로 이뤄진다.
1. optimizer.zero_grad()를 호출해 모델 매개변수의 변화도를 재설정한다. - grad는 매 반복마다 명시적으로 0으로 설정
2. loss.backwards()를 호출해 예측 손실(prediction loss)을 역전파한다. pytorch는 각 매개변수에 대한 손실의 변화도를 저장한다.
3. 변화도를 계산한 뒤 optimizer.step()을 호출해 역전파 단계에서 수집된 변화도로 매개변수를 조정한다.
"""