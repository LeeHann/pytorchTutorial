import torch
import numpy as np

"""
텐서는 배열이나 행렬과 매우 유사한 특수한 자료구조이다.
파이토치는 텐서를 사용해 모델의 입력과 출력, 모델의 매개변수를 부호화 한다.

텐서를 초기화하는 방법은 아래와 같다.
1. 데이터로부터 직접 생성하기
"""
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

"""
2. Numpy 배열로부터 생성하기
"""
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

"""
3. 다른 텐서로부터 생성하기
"""
x_ones = torch.ones_like(x_data)    # x_data의 속성을 유지한다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # x_data의 속성을 덮어쓴다.
print(f"Random Tensor: \n {x_rand} \n")

"""
output:
Ones Tensor: 
 tensor([[1, 1],
        [1, 1]]) 

Random Tensor: 
 tensor([[0.9100, 0.7187],
        [0.5405, 0.6588]]) 
"""
