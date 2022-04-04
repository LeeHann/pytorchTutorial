"""
전 장까지 torch.no_grad()로 학습 가능한 매개변수를 갖는 텐서들을 직접 조작해 모델의 가중치를 갱신했다.
이는 확률적 경사 하강법과 같은 간단한 최적화 알고리즘에서는 부담이 되지 않지만
실제 실경망을 학습할 때는 AdaGrad, RMSProp, Adam 등과 같은 더 정교한 최적화 기법을 사용한다.

pytorch의 optim 패키지는 최적화 알고리즘에 대한 아이디어를 추상화하고
일반적으로 사용하는 최적화 알고리즘의 구현체를 제공한다.

아래에서 모델을 최적화 할 때 optim 패키지의 RMSProp을 사용한다.
"""
# -*- coding: utf-8 -*-
import torch
import math

# 입력값과 출력값을 갖는 텐서들을 생성합니다.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 입력 텐서 (x, x^2, x^3)를 준비합니다.
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# nn 패키지를 사용하여 모델과 손실 함수를 정의합니다.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
# nn.MSELoss(평균 제곱 오차)
loss_fn = torch.nn.MSELoss(reduction='sum')

# optim 패키지를 사용하여 모델의 가중치를 갱신할 optimizer를 정의합니다.
# 여기서는 RMSprop을 사용하겠습니다; optim 패키지는 다른 다양한 최적화 알고리즘을 포함하고 있습니다.
# RMSprop 생성자의 첫번째 인자는 어떤 텐서가 갱신되어야 하는지를 알려줍니다.
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
    # 순전파 단계: 모델에 x를 전달하여 예측값 y를 계산합니다.
    y_pred = model(xx)

    # 손실을 계산하고 출력합니다.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계 전에, optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인) 갱신할
    # 변수들에 대한 모든 변화도(gradient)를 0으로 만듭니다. 이렇게 하는 이유는 기본적으로
    # .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고) 누적되기
    # 때문입니다. 더 자세한 내용은 torch.autograd.backward에 대한 문서를 참조하세요.
    optimizer.zero_grad()

    # 역전파 단계: 모델의 매개변수들에 대한 손실의 변화도를 계산합니다.
    loss.backward()

    # optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
    optimizer.step()


linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

"""
output:

99 533.6102294921875
199 289.5715026855469
299 162.5748748779297
399 86.45413208007812
499 41.967594146728516
599 19.943618774414062
699 11.032347679138184
799 9.173364639282227
899 8.945491790771484
999 8.885900497436523
1099 8.891117095947266
1199 8.914182662963867
1299 8.913580894470215
1399 8.91093921661377
1499 8.91108226776123
1599 8.915746688842773
1699 8.922330856323242
1799 8.923245429992676
1899 8.919933319091797
1999 8.919984817504883
Result: y = -0.00048285070806741714 + 0.8562403321266174 x + -0.00048298999899998307 x^2 + -0.09383095055818558 x^3

Process finished with exit code 0
"""