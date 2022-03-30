import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

"""
데이터를 변형(transform)해 데이터를 조작하고 학습을 적합하게 만들 수 있다.
모든 TorchVision 데이터셋은 transform과 target_transform을 매개변수로 가진다.
transform: 특징(feature)을 변경한다.
target_transform: 정답(label)을 변경한다.

FashionNMIST 특징(feature)은 PIL Image 형식으로 정답이 정수이다.
학습을 하려면 정규화된 텐서의 특징과
원-핫(one-hot:정답만 1로 설정하고 정답이 아닌 분류는 0)으로 부호화된 텐서 형태의 정답이 필요하다.
이런 변형을 위해 ToTensor와 Lambda를 사용한다.
"""

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),   # 특징 transform 을 텐서로
    # 정답 transform을 Lambda를 이용해 0,1로 원-핫 형태로 변형
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

"""
ToTensor()
- PIL Image나 Numpy ndarray를 FloatTensor로 변환하고, 이미지 픽셀의 크기(intensity)값을 [0,1] 범위로 비례하게 scaling 한다.

Lambda()
사용자 정의 람다 함수를 적용한다.
위의 예에서는 정수를 원-핫으로 부호화된 텐서로 바꾼다.
torch.zeros(10, dtype=torch.float)
-> 크기 10짜리 0텐서를 만든다
.scatter_(0, torch.tensor(y), value=1)
-> 주어진 정답 y에 해당하는 인덱스에 value=1 을 할당한다.
"""