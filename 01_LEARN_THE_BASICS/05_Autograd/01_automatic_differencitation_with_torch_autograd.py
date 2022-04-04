"""
신경망 학습의 절차는 아래와 같다.
0. 전제
    - 신경망에는 적응 가능한 가중치와 편향 매개변수가 있고, 이 매개변수가 훈련 데이터에 적응하도록 조정하는 과정을 학습이라한다.
1. 미니배치
    - 훈련 데이터 중 일부를 무작위로 가져온다. 이 선별 데이터를 미니배치라 하며, 미니배치의 손실함수 값을 줄이는 것이 목표이다.
2. 기울기 산출
    - 미니배치의 손실함수 값을 줄이기위해 각 가중치 매개변수의 기울기를 구한다. 기울기는 손실함수를 가장 작게하는 방향으로 나아간다.
3. 매개변수 갱신
    - 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.
4. 반복
    - 1~3단계를 반복한다.

이 학습 절차에서 매개변수를 갱신하기 위해 사용하는 방법으로 역전파 알고리즘이 가장 자주 사용된다.
이 알고리즘에서 매개변수는 주어진 매개변수에 대한 손실함수의 변화도(gradient)에 따라 조정된다.

pytorch는 torch.autograd 라는 자동 미분 엔진이 내장되어 있다.
이를 실습해보자.
"""

import torch

# 입력 x, 매개변수 w, b, 손실함수 교차엔트로피오차 함수
# w, b는 최적화해야하는 매개변수로 이 변수에 대한 손실함수의 변화도를 계산할 수 있어야하기 때문에 requires_grad=True로 한다.
# requires_grad의 값은 텐서를 생성할 때 설정하거나, 나중에 x.requires_grad_(True) 메소드를 사용해 나중에 설정할 수도 있다.
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 연산 그래프 역할 함수 grad_fn(순전파 방향 함수 계산 방법, 역방향 전파 단계 도함수 계산 방법)
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

"""
output:

Gradient function for z = <AddBackward0 object at 0x0000021AD2B51190>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x0000021AD2B51190>
"""

# 매개변수 가중치 최적화를 위한 손실함수의 도함수 계산
# 연산 그래프의 잎 노드 중 requires_grad=True 인 노드의 grad 속성만 구할 수 있음
loss.backward()
print(w.grad)
print(b.grad)
# 성능 상 이유로 그래프에서 backward를 사용한 변화도 계산(grad)은 한번만 수행 가능
# 여러번의 backward 호출이 필요한 경우 backward 호출 시 retrain_graph=True 를 전달해야한다.

"""
output:

tensor([[0.2085, 0.3059, 0.2351],
        [0.2085, 0.3059, 0.2351],
        [0.2085, 0.3059, 0.2351],
        [0.2085, 0.3059, 0.2351],
        [0.2085, 0.3059, 0.2351]])
tensor([0.2085, 0.3059, 0.2351])
"""

"""
기본적으로 requires_grad=True인 모든 텐서는 연산 기록을 추적하고 변화도 계산(grad)을 지원한다.
이러한 추적, 변화도 계산 지원이 필요없을 경우(학습 완료 후 입력데이터를 단순히 적용하는 순전파 연산만 필요한 경우)
torch.no_grad() 블록으로 둘러싸 연산 추적을 멈출 수 있다.
"""

z = torch.matmul(x, w)+b
print(z.requires_grad)  # True

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)  # False

# torch.no_grad()와 동일한 다른 방법 - detach()
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)  # False
