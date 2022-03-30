import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
신경망은 데이터에 대한 연산을 수행하는 계층으로 구성되어 있다.
신경망은 다른 모듈로 구성된 모듈로 중첩된 구조로 인해 복잡한 아키텍처를 쉽게 구축, 관리할 수 있다.
torch.nn은 신경망을 구성하는데 필요한 모든 구성요소를 제공한다.

아래는 FashionMNIST 데이터셋의 이미지를 분류하는 신경망이다.
"""

# 학습을 위한 장치 얻기
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

"""
output:
Using cpu device
"""


# 클래스 정의
class NeuralNetwork(nn.Module):  # nn.Module 을 부모 클래스로 사용
    def __init__(self):
        # super - 부모 클래스로 초기화
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # nn 의 Flatten 함수 사용
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # 순전파
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# NeuralNetwork의 인스턴스를 생성하고 이를 device로 이동
model = NeuralNetwork().to(device)
# model structure 출력
print(model)

"""
output:
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)

"""

# 입력 데이터 X
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
# 원시 예측값을 softmax에 통과시켜 예측 확률을 얻는다.
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

"""
output:
Predicted class: tensor([7])
"""

# 미니배치 생성 (이미지 28*28 사이즈의 이미지 3개로 구성)
input_image = torch.rand(3,28,28)
print(input_image.size())

"""
output:
torch.Size([3, 28, 28])
"""

# nn.Flatten 은 input_image를 평탄화=1차원으로 만들어서
# 28*28 배열을 784 픽셀 값을 갖는 연속된 배열로 변환
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

"""
output:
torch.Size([3, 784])
"""

# nn.Linear 선형 계층은 저장된 가중치(weight)와 편향(bias)을 사용해 선형 변환(Linear Transform)
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

"""
output:
torch.Size([3, 20])
"""

# nn.ReLU 비선형 활성화 함수로 모델의 입력과 출력 사이에 복잡한 관계를 만듦
# x > 0 -> x 출력, x <= 0 -> 0 출력
# 비선형 활성화는 선형 변환 후에 적용되어 비선형성을 도입하고 신경망의 다양한 현상 학습을 돕는다.
# 다른 비선형 활성화 함수도 존재
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

"""
output:
Before ReLU: tensor([[-2.0177e-01,  1.9768e-01,  4.4918e-02,  2.9432e-02,  1.5237e-02,
         -2.7254e-01, -2.4435e-01,  3.4354e-01, -2.2732e-01,  9.1682e-02,
          3.4947e-02, -2.3257e-01,  6.8991e-02, -2.6522e-01, -3.0311e-01,
         -1.6716e-02,  2.4998e-01, -1.3499e-01,  4.9211e-02, -1.6438e-01],
        [ 1.4455e-03, -8.4558e-02, -1.0713e-01, -1.4143e-01, -9.3926e-02,
         -1.1508e-01, -4.7018e-01,  4.7610e-01, -3.0134e-02,  1.6504e-01,
         -4.1265e-01, -2.0290e-01, -1.2494e-01, -1.6368e-01, -1.2779e-01,
          2.6979e-02,  4.7788e-01, -3.6697e-03,  2.2985e-01, -5.1763e-01],
        [ 2.6800e-04,  2.5453e-01, -8.3859e-02, -1.8699e-01, -3.0041e-01,
          6.3344e-04, -4.8898e-01,  2.8160e-01, -7.0366e-02, -1.9719e-01,
          3.7933e-01, -1.4543e-01, -3.1273e-01, -2.1764e-01, -5.8061e-01,
          2.7932e-03,  4.3528e-01,  4.6559e-02,  2.9417e-01, -3.5216e-01]],
       grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.0000e+00, 1.9768e-01, 4.4918e-02, 2.9432e-02, 1.5237e-02, 0.0000e+00,
         0.0000e+00, 3.4354e-01, 0.0000e+00, 9.1682e-02, 3.4947e-02, 0.0000e+00,
         6.8991e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4998e-01, 0.0000e+00,
         4.9211e-02, 0.0000e+00],
        [1.4455e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 4.7610e-01, 0.0000e+00, 1.6504e-01, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 2.6979e-02, 4.7788e-01, 0.0000e+00,
         2.2985e-01, 0.0000e+00],
        [2.6800e-04, 2.5453e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.3344e-04,
         0.0000e+00, 2.8160e-01, 0.0000e+00, 0.0000e+00, 3.7933e-01, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 2.7932e-03, 4.3528e-01, 4.6559e-02,
         2.9417e-01, 0.0000e+00]], grad_fn=<ReluBackward0>)
"""

# nn.Sequential 은 순서를 갖는, 모듈 컨테이너로, 데이터가 정의된 순서로 전달
# 순차 컨테이너를 사용해 seq_modules 같은 신경망을 빠르게 만들 수 있다.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# nn.Softmax 는 신경망의 마지막 선형 계층 뒤에 쓰며,
# 원시값을 각 분류에 대한 예측 확률([0,1])로 조정한다.
# dim 매개변수는 값의 합이 1이 되는 차원을 가리킨다.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# 모델 매개변수
# 신경망 내에 많은 계층이 매개변수화(가중치, 편향과 연관지어지기) 되는데,
# nn.Module 을 상속하면 모델 객체 내부의 모든 필드들이 자동으로 track 되며,
# model의 parameters() 및 named_parameters() 메소드로 모든 매개변수에 접근할 수 있게 된다.
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

"""
output:
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0041,  0.0060, -0.0155,  ..., -0.0248, -0.0098, -0.0100],
        [ 0.0046,  0.0058,  0.0013,  ...,  0.0307,  0.0347,  0.0347]],
       grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([0.0229, 0.0180], grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0403, -0.0030, -0.0310,  ..., -0.0190, -0.0232, -0.0181],
        [ 0.0131, -0.0127, -0.0434,  ...,  0.0011, -0.0405,  0.0408]],
       grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0371, -0.0080], grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0440,  0.0297,  0.0219,  ...,  0.0175,  0.0189,  0.0077],
        [-0.0438,  0.0290,  0.0330,  ..., -0.0439,  0.0304, -0.0204]],
       grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0041, -0.0042], grad_fn=<SliceBackward0>) 

"""