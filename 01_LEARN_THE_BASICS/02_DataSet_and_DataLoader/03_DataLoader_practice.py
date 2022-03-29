from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
- DataLoader로 학습용 데이터 준비하기
Dataset은 데이터셋의 특징(feature)을 가져오고 하나의 샘플에 정답을 지정하는 일을 한 번에 한다.
모델을 학습할 때, 1) 샘플을 미니배치(minibatch)로 전달하고 
2) 매 에폭(epoch)마다 데이터를 다시 섞어 오버피팅을 막고,
3) python의 multiprocessing을 써서 데이터 검색 속도를 높인다.

DataLoader는 간단한 API로 위 과정을 추상화한 순회 가능한 객체이다.
"""
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# 미니배치 사이즈는 batch_size=64 이다.
# shuffle=True로 매 epoch 후 데이터가 섞이며 오버피팅을 막는다.

"""
- DataLoader를 통해 순회하기

아래의 각 순회(iteration)는 train_features와 train_labels의 묶음을 반환한다.
"""

# 이미지와 정답(label)을 반환한다.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
