import torch

"""
Numpy 식의 표준 인덱싱과 슬라이싱 []
"""

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

"""
output:
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

"""

"""
텐서의 연산에는 
전치(transposing), 인덱싱(indexing), 슬라이싱(slicing), 수학 계산, 선형 대수, 임의 샘플링(random sampling) 등
여러 가지 연산이 있다.
각 연산은 GPU에서 실행할 수 있는데,
기본적으로 텐서는 CPU에 생성되고 .to 메소드를 사용하면 GPU로 텐서를 명시적으로 이동해 사용할 수 있다.
다만 장치들 간에 큰 텐서들을 복사하는 것은 시간과 메모리 측면에서 비용이 많이 든다.
"""

# GPU가 존재하면 텐서를 이동한다.
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


"""
텐서 합치기
torch.cat 을 사용하면 주어진 차원에 따라 일련의 텐서를 연결할 수 있다.
텐서 결합 연산에는 torch.stack도 있다.
"""

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"t1 : {t1}")

"""
output:
t1 : tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

"""

"""
산술 연산
"""

# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산한다.
# y1, y2, y3은 모두 같은 값을 갖는다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

print(f"y1 : {y1} \ny2 : {y2} \ny3 : {y3}\n")

# 요소별 곱(element-wise product)을 계산한다.
# z1, z2, z3는 모두 같은 값을 갖는다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(f"z1 : {z1}\nz2 : {z2}\nz3 : {z3}")

"""
output:
y1 : tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]]) 
y2 : tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]]) 
y3 : tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

z1 : tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
z2 : tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
z3 : tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
"""

"""
단일-요소(single-element) 텐서:
텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 텐서의 경우
item()을 사용해 python 숫자 값으로 변환할 수 있다.
"""

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

"""
output:
12.0 <class 'float'>
"""

"""
바꿔치기(in-place) 연산:
연산 결과를 피연산자(operand)에 저장하는 연산으로 '_' 접미사를 갖는다.
x.copy_(y) 나 x.t_() 는 x 를 변경한다.

바꿔치기 연산은 메모리를 일부 절약하지만, 
기록(history)이 즉시 삭제되어 도함수(derivative) 계산에 문제가 발생할 수 있다. 
따라서, 사용을 권장하지 않는다.
"""

print(f"in-place before : {tensor} \n")
tensor.add_(5)
print(f"in-place after : {tensor}\n")

"""
in-place before : tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 

in-place after : tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])

"""