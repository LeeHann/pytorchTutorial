"""
numpy는 과학 분야의 연산을 위한 포괄적인 프레임워크로 연산 그래프나 딥러닝, 변화도를 제공하지는 않지만
신경망의 순전파 단계와 역전파 단계를 직접 구현함으로써 3차 다항식이 사인 함수에 근사하도록 만들 수 있다.
"""

# -*- coding: utf-8 -*-
import numpy as np
import math

# 무작위로 입력과 출력 데이터를 생성합니다
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# 무작위로 가중치를 초기화합니다
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):  # epoch = 2000
    # 순전파 단계: 예측값 y를 계산한다
    # y = a + bx + cx² + dx³
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 손실(loss)을 계산하고 출력한다
    # 오차제곱합
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # 손실에 따른 a, b, c, d의 변화도(gradient)를 계산하고 역전파한다.
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

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

"""
output:

99 911.369391987195
199 627.1870523939637
299 432.97034234564427
399 300.0844906228241
499 209.05754643876114
599 146.6324991326724
699 103.77340457980713
799 74.31435799702442
899 54.043073066029024
999 40.078584275146206
1099 30.448207677186844
1199 23.799636385490338
1299 19.204782325428738
1399 16.0259753648157
1499 13.824592314783919
1599 12.298589074662534
1699 11.239743933652043
1799 10.5043587263513
1899 9.993158934353882
1999 9.637488844876081
Result: y = -0.02700956071684126 + 0.8693855156754136 x + 0.004659598664685429 x^2 + -0.09512898360740282 x^3
"""