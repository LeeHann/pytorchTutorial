"""
가중치 공유 : CNN에서 가중치 공유란, 하나의 커널이 뉴런의 볼륨을 stride 해서 모든 커널이 동일한 가중치를 갖는 것을 의미한다.
또한 이러한 가중치, 즉 볼륨을 지나며 모든 커널이 학습하는 동안에도 동일하게 유지됨을 의미한다.
[https://i-am-eden.tistory.com/30]

"""

# -*- coding: utf-8 -*-
import random
import torch
import math


class DynamicNet(torch.nn.Module):
    def __init__(self):
        """
        생성자에서 5개의 매개변수를 생성(instantiate)하고 멤버 변수로 지정합니다.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        모델의 순전파 단계에서는 무작위로 4, 5 중 하나를 선택한 뒤 매개변수 e를 재사용하여
        이 차수들의의 기여도(contribution)를 계산합니다.

        각 순전파 단계는 동적 연산 그래프를 구성하기 떄문에, 모델의 순전파 단계를 정의할 때
        반복문이나 조건문과 같은 일반적인 Python 제어-흐름 연산자를 사용할 수 있습니다.

        여기에서 연산 그래프를 정의할 때 동일한 매개변수를 여러번 사용하는 것이 완벽히 안전하다는
        것을 알 수 있습니다.
        """
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        """
        Python의 다른 클래스(class)처럼, PyTorch 모듈을 사용해서 사용자 정의 메소드를 정의할 수 있습니다.
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'


# 입력값과 출력값을 갖는 텐서들을 생성합니다.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 위에서 정의한 클래스로 모델을 생성합니다.
model = DynamicNet()

# 손실 함수와 optimizer를 생성합니다. 이 이상한 모델을 순수한 확률적 경사하강법(SGD; Stochastic Gradient Descent)으로
# 학습하는 것은 어려우므로, 모멘텀(momentum)을 사용합니다.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(30000):
    # 순전파 단계: 모델에 x를 전달하여 예측값 y를 계산합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력합니다.
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())

    # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')

"""
output:

1999 2144.73388671875
3999 1013.8048706054688
5999 453.7138366699219
7999 196.87135314941406
9999 95.2450180053711
11999 47.80988311767578
13999 25.592472076416016
15999 16.189292907714844
17999 12.04482650756836
19999 10.230865478515625
21999 9.46541976928711
23999 9.086564064025879
25999 8.965790748596191
27999 8.680519104003906
29999 8.572198867797852
Result: y = 0.0018837854731827974 + 0.8528687357902527 x + -0.0008350106654688716 x^2 + -0.09321363270282745 x^3 + 0.00014122006541583687 x^4 ? + 0.00014122006541583687 x^5 ?
"""