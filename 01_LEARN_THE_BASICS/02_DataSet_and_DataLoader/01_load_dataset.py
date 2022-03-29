import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

"""
데이터셋 코드는 모델 학습 코드로부터 분리하는 것이 가독성과 모듈성 측면에서 좋다.
파이토치는 이 점에서 torch.utils.data.DataLoader 와 torch.utils.data.Dataset 의 모듈을 제공한다.
Dataset : 샘플과 정답(label)을 저장한다.
DataLoader : Dataset을 샘플에 쉽게 접근할 수 있도록 순회하는 객체(iterable)로 감싼다.

TorchVision 에서 Fashion-MNIST 데이터셋을 불러오는 예제를 살펴본다.
Fashion-MNIST 데이터셋의 각 예제는 흑백의 28*28 이미지와 10개 분류(class) 중 하나인 정답(label)로 구성된다.

root: 학습/테스트 데이터가 저장되는 경로
train: 학습용 또는 테스트용 데이터셋 여부를 지정
download=True: root 에 데이터가 없는 경우 인터넷에서 다운로드
transform, target_transform: 특징(feature)과 정답(label) 변형(transform)을 지정
"""

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

"""
training_data[index]로 Dataset에 리스트처럼 직접 접근(=index)할 수 있다.
"""

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
