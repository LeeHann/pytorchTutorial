import os
import pandas as pd
from torchvision.io import read_image

"""
사용자 정의 dataset 클래스는 아래의 3개 함수를 구현해야한다.
__init__ : 객체가 생성될 때 한 번만 실행되는 생성자. 이미지와 주석 파일(annotation_file)이 포함된
디렉토리와 두가지 변형(transform)을 초기화한다.
__len__ : 데이터셋의 샘플 개수를 반환한다.
__getitem__ : 주어진 인덱스 idx에 해당하는 샘플을 데이터셋에서 불러오고 반환한다. get

아래 구현에서 FashionMNIST 이미지들은 img_dir 디렉토리에 저장되고 정답은 annotations_file csv 파일에 별도로 저장된다.
"""


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):  # idx로 이미지 위치를 식별한다.
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)    # 이미지를 텐서로 변환한다
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label     # 이미지와 레이블을 반환한다.

