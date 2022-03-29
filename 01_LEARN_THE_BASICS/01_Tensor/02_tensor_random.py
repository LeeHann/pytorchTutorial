import torch
import numpy as np

"""
무작위(random) 또는 상수(constant) 값을 사용하기
shape 은 텐서의 차원을 나타내는 튜플로, 아래 함수에서 출력 텐서의 차원을 결정한다.
"""

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

"""
output:
Random Tensor: 
 tensor([[0.3011, 0.4590, 0.3814],
        [0.3370, 0.3673, 0.9542]]) 

Ones Tensor: 
 tensor([[1., 1., 1.],
        [1., 1., 1.]]) 

Zeros Tensor: 
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
"""