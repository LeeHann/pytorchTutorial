import torch
import torchvision.models as models

"""
모델을 저장 및 불러오기를 통해 상태를 유지(persist)하고 모델이 예측을 실행하는 방법을 알아보자.

pytorch 모델은 학습한 매개변수를 state_dict 라고 불리는 내부 상태 사전에 저장한다.
이 상태값들은 torch.save 메소드를 사용해 저장할 수 있다.
또한 불러오기는 먼저 동일한 인스턴스를 생성해 load_state_dict() 메소드를 사용해 불러올 수 있다.
"""
# save
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# load
model = models.vgg16() # 기본 가중치를 불러오지 않으므로 pretrained=True를 지정하지 않습니다.
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # 추론을 하기 전 드롭아웃과 배치 정규화를 평가 모드로 설정한다. 그렇지 않으면 일관성없는 추론 결과 생성

"""
모델의 형태를 포함해 저장하고 불러오는 방법은 아래와 같다.
이 접근 방식은 Python pickle 모듈을 사용하여 모델을 직렬화(serialize)하므로, 
모델을 불러올 때 실제 클래스 정의(definition)를 적용(rely on)한다.
"""
# 클래스 구조를 모델과 함께 저장하기
torch.save(model, 'model.pth')

# 불러오기
model = torch.load('model.pth')

