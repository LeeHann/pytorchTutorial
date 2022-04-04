"""
기존 nn.Module 의 구성(sequence)보다 더 복잡한 모델을 구성해야할 때 사용한다.
이때는 nn.Module의 하위클래스로 새로운 Module을 정의하고,
forward를 정의한다.

아래에서 3차 다항식을 사용자 정의 Module 하위 클래스로 구현한다.
"""

# -*- coding: utf-8 -*-
import torch
import math


class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        생성자에서 4개의 매개변수를 생성(instantiate)하고, 멤버 변수로 지정합니다.
        """
        super().__init__()  # 상위의 torch.nn.Module을 상속받는다
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        순전파 함수에서는 입력 데이터의 텐서를 받고 출력 데이터의 텐서를 반환해야 합니다.
        텐서들 간의 임의의 연산뿐만 아니라, 생성자에서 정의한 Module을 사용할 수 있습니다.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Python의 다른 클래스(class)처럼, PyTorch 모듈을 사용해서 사용자 정의 메소드를 정의할 수 있습니다.
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'


# 입력값과 출력값을 갖는 텐서들을 생성합니다.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 위에서 정의한 클래스로 모델을 생성합니다.
model = Polynomial3()

# 손실 함수와 optimizer를 생성합니다. SGD 생성자에 model.paramaters()를 호출해주면
# 모델의 멤버 학습 가능한 (torch.nn.Parameter로 정의된) 매개변수들이 포함됩니다.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    # 순전파 단계: 모델에 x를 전달하여 예측값 y를 계산합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력합니다.
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')

"""
output:

99 1485.181640625
199 1053.4617919921875
299 748.0018310546875
399 531.86279296875
499 378.9252014160156
599 270.7076416015625
699 194.133056640625
799 139.9488525390625
899 101.6078109741211
999 74.47740173339844
1099 55.27959442138672
1199 41.69499206542969
1299 32.08230972290039
1399 25.280168533325195
1499 20.46685028076172
1599 17.06083869934082
1699 14.65064525604248
1799 12.945128440856934
1899 11.738252639770508
1999 10.884232521057129
Result: y = -0.04811493307352066 + 0.8576081395149231 x + 0.008300626650452614 x^2 + -0.09345375001430511 x^3
"""