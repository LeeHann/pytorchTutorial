"""
텐서는 종종 50배 혹은 그 이상의 속도 향상을 제공하는 GPU를 사용할 수 있다.
텐서는 개념적으로 numpy 배열과 동일하지만,
pytorch가 텐서의 다양한 연산을 위해 다양항 기능을 제공한다.
또한 numpy와 달리 GPU를 사용해 수치 연산을 가속할 수 있다.
pytorch 텐서를 gpu에서 실행하기 위해서는 단지 적절한 장치를 지정해주기만 하면 된다.
# device = torch.device("cuda:0") 이렇게.
"""

# -*- coding: utf-8 -*-

import torch
import math


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요

# 무작위로 입력과 출력 데이터를 생성합니다
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 무작위로 가중치를 초기화합니다
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계: 예측값 y를 계산합니다
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 손실(loss)을 계산하고 출력합니다
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # 손실에 따른 a, b, c, d의 변화도(gradient)를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 가중치를 갱신합니다.
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

"""
output:

99 3531.076904296875
199 2469.493896484375
299 1729.0599365234375
399 1212.224365234375
499 851.2020263671875
599 598.843505859375
699 422.32421875
799 298.77392578125
899 212.2452850341797
999 151.6094970703125
1099 109.0947265625
1199 79.269775390625
1299 58.33659744262695
1399 43.63713455200195
1499 33.31047439575195
1599 26.052610397338867
1699 20.94952392578125
1799 17.36012840270996
1899 14.834478378295898
1999 13.056734085083008
Result: y = -0.06672961264848709 + 0.8409093618392944 x + 0.01151196751743555 x^2 + -0.0910784974694252 x^3

"""