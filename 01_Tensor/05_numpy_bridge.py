import torch
import numpy as np

"""
CPU 상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에, 
하나를 변경하면 다른 하나도 변경된다.
"""

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

"""
output:
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
"""

# 텐서의 변경 사항이 Numpy 배열에 반영된다.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

"""
output:
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
"""

# numpy 배열을 텐서로 변환한다.
n = np.ones(5)
t = torch.from_numpy(n)

# numpy 배열의 변경 사항이 텐서에 반영된다.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

"""
output:
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
"""