"""
연산그래프 = 계산그래프 (Computational graph)
autograd는 데이터의 실행된 모든 연산의 기록을
Function 객체로 구성된 방향성 비순환 그래프(DAG; Directed Acyclic Graph)에 저장한다.
이 DAG의 잎(leave)은 입력 텐서이고, 뿌리(root)는 결과 텐서이다.
이 그래프의 뿌리에서 잎까지 추적하면 연쇄법칙에 따라 변화도를 자동으로 계산할 수 있다.

순전파 단계에서 autograd는 두 가지 작업을 동시에 수행한다.
1. 요청된 연산을 수행해 결과 텐서를 계산하기
2. DAG에 연산의 변화도 기능(gradient function)을 유지한다.

역전파 단계는 DAG 뿌리에서 .backward()가 호출될 때 시작된다.
이때 autograd는
.grad_fn 에서 변화도(grad)를 계산하고,
각 텐서의 .grad 속성에 계산 결과를 쌓고(accumulate)
연쇄 법칙을 이용해, 모든 잎(leaf) 텐서까지 전파한다.

- Pytorch에서 DAG는 동적(dynamic)이다.
그래프가 처음부터 다시 생성된다.
그래프가 backward가 호출되고 나면 autograd는 새로운 그래프를 채우기 시작하고,
이 점 때문에 모델에서 흐름 제어 구문을 사용할 수 있게 되는 것이다.
매번 반복할 때마다 모양(shape), 크기(size), 연산(operation)을 바꿀 수 있다.
"""

"""
[선택적으로 읽기(Optional Reading): 텐서 변화도와 야코비안 곱(Jacobian Product)]

출력함수가 임의의 텐서인 경우 실제 변화도가 아닌 야코비안 곱을 계산한다.
야코비안 행렬 자체를 계산하는 대신, 주어진 입력 벡터 v 에 대한 야코비안 곱을 계산하는데 
이 과정은 v를 인자로 backward 호출을 하면 이뤄진다.
v의 크기는 곱을 계산하려고 하는 원래 텐서의 크기와 같아야한다.
"""
import torch

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

"""
output:

First call
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.],
        [2., 2., 2., 2., 4.]])

Second call
tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.],
        [4., 4., 4., 4., 8.]])

Call after zeroing gradients
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.],
        [2., 2., 2., 2., 4.]])
"""

"""
동일한 인자로 backward를 두차례 호출하면 변화도 값이 달라진다.
역전파 수행시 pytorch가 변화도를 누적하기 때문에 계산된 변화도의 값이 연산 그래프의 모든 잎 노드의 grad 속성에 추가된다.
따라서 제대로된 변화도를 계산하기 위해 grad 속성을 먼저 0으로 만들어야 한다.
이 과정을 실제로는 옵티마이저(optimizer)가 도와준다.
"""