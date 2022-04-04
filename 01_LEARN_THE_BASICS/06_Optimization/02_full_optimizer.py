import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model = NeuralNetwork()
learning_rate = 1e-3
batch_size = 64
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

"""
손실 함수와 옵티마이저를 초기화하고 train_loop와 test_loop에 전달합니다. 
모델의 성능 향상을 알아보기 위해 자유롭게 에폭(epoch) 수를 증가시켜 볼 수 있다.

output:
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data\FashionMNIST\raw\train-images-idx3-ubyte.gz
100.0%
Extracting data\FashionMNIST\raw\train-images-idx3-ubyte.gz to data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data\FashionMNIST\raw\train-labels-idx1-ubyte.gz
100.6%
Extracting data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz
100.0%
Extracting data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
Extracting data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to data\FashionMNIST\raw

119.3%
Epoch 1
-------------------------------
loss: 2.290805  [    0/60000]
loss: 2.285262  [ 6400/60000]
loss: 2.267396  [12800/60000]
loss: 2.272673  [19200/60000]
loss: 2.258700  [25600/60000]
loss: 2.213551  [32000/60000]
loss: 2.238148  [38400/60000]
loss: 2.192095  [44800/60000]
loss: 2.186999  [51200/60000]
loss: 2.169276  [57600/60000]
Test Error: 
 Accuracy: 39.3%, Avg loss: 2.160595 

Epoch 2
-------------------------------
loss: 2.166515  [    0/60000]
loss: 2.162455  [ 6400/60000]
loss: 2.108738  [12800/60000]
loss: 2.124144  [19200/60000]
loss: 2.086561  [25600/60000]
loss: 2.018839  [32000/60000]
loss: 2.050908  [38400/60000]
loss: 1.967874  [44800/60000]
loss: 1.967922  [51200/60000]
loss: 1.912611  [57600/60000]
Test Error: 
 Accuracy: 57.5%, Avg loss: 1.905111 

Epoch 3
-------------------------------
loss: 1.939521  [    0/60000]
loss: 1.913741  [ 6400/60000]
loss: 1.798219  [12800/60000]
loss: 1.828289  [19200/60000]
loss: 1.741905  [25600/60000]
loss: 1.681661  [32000/60000]
loss: 1.702456  [38400/60000]
loss: 1.597232  [44800/60000]
loss: 1.620226  [51200/60000]
loss: 1.525670  [57600/60000]
Test Error: 
 Accuracy: 60.0%, Avg loss: 1.536658 

Epoch 4
-------------------------------
loss: 1.606153  [    0/60000]
loss: 1.571635  [ 6400/60000]
loss: 1.417832  [12800/60000]
loss: 1.484226  [19200/60000]
loss: 1.381420  [25600/60000]
loss: 1.362128  [32000/60000]
loss: 1.382668  [38400/60000]
loss: 1.298802  [44800/60000]
loss: 1.335680  [51200/60000]
loss: 1.247083  [57600/60000]
Test Error: 
 Accuracy: 62.5%, Avg loss: 1.265909 

Epoch 5
-------------------------------
loss: 1.345512  [    0/60000]
loss: 1.324851  [ 6400/60000]
loss: 1.157245  [12800/60000]
loss: 1.258405  [19200/60000]
loss: 1.144803  [25600/60000]
loss: 1.157681  [32000/60000]
loss: 1.184801  [38400/60000]
loss: 1.118479  [44800/60000]
loss: 1.156882  [51200/60000]
loss: 1.084893  [57600/60000]
Test Error: 
 Accuracy: 63.9%, Avg loss: 1.098492 

Epoch 6
-------------------------------
loss: 1.172832  [    0/60000]
loss: 1.169971  [ 6400/60000]
loss: 0.989187  [12800/60000]
loss: 1.116778  [19200/60000]
loss: 1.002002  [25600/60000]
loss: 1.022938  [32000/60000]
loss: 1.062161  [38400/60000]
loss: 1.004976  [44800/60000]
loss: 1.041378  [51200/60000]
loss: 0.982642  [57600/60000]
Test Error: 
 Accuracy: 65.2%, Avg loss: 0.991096 

Epoch 7
-------------------------------
loss: 1.054187  [    0/60000]
loss: 1.070105  [ 6400/60000]
loss: 0.875206  [12800/60000]
loss: 1.023347  [19200/60000]
loss: 0.914987  [25600/60000]
loss: 0.928797  [32000/60000]
loss: 0.982023  [38400/60000]
loss: 0.931208  [44800/60000]
loss: 0.961417  [51200/60000]
loss: 0.914297  [57600/60000]
Test Error: 
 Accuracy: 66.5%, Avg loss: 0.918257 

Epoch 8
-------------------------------
loss: 0.967568  [    0/60000]
loss: 1.001492  [ 6400/60000]
loss: 0.794483  [12800/60000]
loss: 0.957771  [19200/60000]
loss: 0.857967  [25600/60000]
loss: 0.860649  [32000/60000]
loss: 0.925890  [38400/60000]
loss: 0.881946  [44800/60000]
loss: 0.903640  [51200/60000]
loss: 0.865390  [57600/60000]
Test Error: 
 Accuracy: 67.6%, Avg loss: 0.865953 

Epoch 9
-------------------------------
loss: 0.901187  [    0/60000]
loss: 0.950740  [ 6400/60000]
loss: 0.734297  [12800/60000]
loss: 0.909306  [19200/60000]
loss: 0.817288  [25600/60000]
loss: 0.809292  [32000/60000]
loss: 0.883764  [38400/60000]
loss: 0.847281  [44800/60000]
loss: 0.860194  [51200/60000]
loss: 0.827662  [57600/60000]
Test Error: 
 Accuracy: 68.9%, Avg loss: 0.826268 

Epoch 10
-------------------------------
loss: 0.848096  [    0/60000]
loss: 0.910417  [ 6400/60000]
loss: 0.687048  [12800/60000]
loss: 0.871887  [19200/60000]
loss: 0.786192  [25600/60000]
loss: 0.769716  [32000/60000]
loss: 0.850085  [38400/60000]
loss: 0.821432  [44800/60000]
loss: 0.826244  [51200/60000]
loss: 0.797040  [57600/60000]
Test Error: 
 Accuracy: 70.2%, Avg loss: 0.794726 

Done!
"""