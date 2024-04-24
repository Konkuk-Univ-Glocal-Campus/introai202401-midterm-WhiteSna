  
# Script file to hide implementation details for PyTorch computer vision module

import builtins
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import zipfile 

# GPU(CUDA)를 사용하거나 없다면 CPU를 사용
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load_mnist data를 가져오는 함수를 정의하고 Fashion_mnist 데이터 셋을 불러오도록 저장
def load_mnist(batch_size=64): 
    builtins.data_train = torchvision.datasets.FashionMNIST('./data',
        download=True,train=True,transform=ToTensor()) 
    builtins.data_test = torchvision.datasets.FashionMNIST('./data', 
        download=True,train=False,transform=ToTensor()) 
    builtins.train_loader = torch.utils.data.DataLoader(data_train,batch_size=batch_size) 
    builtins.test_loader = torch.utils.data.DataLoader(data_test,batch_size=batch_size)

# train_epoch 함수 정의
# 신경망을 한 에폭(epoch) 동안 학습하는 과정을 구현한 Python 함수
# 이 함수는 모델을 학습시키고, 각 배치에서의 평균 손실과 정확도를 계산하여 반환하는데 이를 통해 학습 과정을 모니터링할 수 있음
def train_epoch(net,dataloader,lr=0.01,optimizer=None,loss_fn = nn.NLLLoss()): 
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr) 
    net.train()                          
    total_loss,acc,count = 0,0,0 
    for features,labels in dataloader: 
        optimizer.zero_grad()                           # optimizer를 초기화 시킨 후 새로운 학습을 시작
        lbls = labels.to(default_device) 
        out = net(features.to(default_device)) 
        loss = loss_fn(out,lbls) 
        loss.backward()                                 # 기울기 계산
        optimizer.step() 
        total_loss+=loss                                # total_loss 값 계산
        _,predicted = torch.max(out,1) 
        acc+=(predicted==lbls).sum() 
        count+=len(labels) 
    return total_loss.item()/count, acc.item()/count    # loss와 acc를 출력

# 주어진 신경망 모델을 평가하는 과정을 나타내는 Python 함수
# 이 함수는 주어진 데이터 로더를 사용하여 모델의 성능을 평가하고, 평균 손실과 정확도를 반환하여 모델의 효율성을 확인
def validate(net, dataloader,loss_fn=nn.NLLLoss()): 
    net.eval()
    count,acc,loss = 0,0,0 
    with torch.no_grad(): 
        for features,labels in dataloader: 
            lbls = labels.to(default_device) 
            out = net(features.to(default_device)) 
            loss += loss_fn(out,lbls) 
            pred = torch.max(out,1)[1] 
            acc += (pred==lbls).sum() 
            count += len(labels) 
    return loss.item()/count, acc.item()/count 

# 신경망 모델을 여러 에폭(epoch) 동안 학습하고 평가하는 과정을 정의하는 Python 함수

def train(net,train_loader,test_loader,optimizer=None,lr=0.01,epochs=10,loss_fn=nn.NLLLoss()): 
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr) 
    res = { 'train_loss' : [], 'train_acc': [], 'val_loss': [], 'val_acc': []} 
    for ep in range(epochs): # 지정된 에폭 수만큼 반복
        tl,ta = train_epoch(net,train_loader,optimizer=optimizer,lr=lr,loss_fn=loss_fn) 
        vl,va = validate(net,test_loader,loss_fn=loss_fn) 
        print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}") 
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
    return res # 학습과 검증 과정에서의 결과를 담은 딕셔너리를 반환

# 신경망 모델을 학습하면서 주기적으로 학습 상태를 출력하고, 각 에폭의 끝에서 검증 성능을 출력하는 Python 함수
def train_long(net,train_loader,test_loader,epochs=5,lr=0.01,optimizer=None,loss_fn = nn.NLLLoss(),print_freq=10):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    for epoch in range(epochs):
        net.train()
        total_loss,acc,count = 0,0,0
        for i, (features,labels) in enumerate(train_loader):
            lbls = labels.to(default_device)
            optimizer.zero_grad()
            out = net(features.to(default_device))
            loss = loss_fn(out,lbls)
            loss.backward()
            optimizer.step()
            total_loss+=loss
            _,predicted = torch.max(out,1)
            acc+=(predicted==lbls).sum()
            count+=len(labels)
            if i%print_freq==0:
                print("Epoch {}, minibatch {}: train acc = {}, train loss = {}".format(epoch,i,acc.item()/count,total_loss.item()/count))
        vl,va = validate(net,test_loader,loss_fn)
        print("Epoch {} done, validation acc = {}, validation loss = {}".format(epoch,va,vl))

# 학습의 결과 값을 나타내는 함수
def plot_results(hist):
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')       # hist 딕셔너리에서 학습 정확도(train_acc)를 추출하여 그래프로 그리는데 라벨을 'Training acc'로 지정하여 그래프에 범례를 추가
    plt.plot(hist['test_acc'], label='Testing acc')         # hist 딕셔너리에서 검증 정확도(test_acc)를 추출하여 그래프로 그리는데 라벨을 'Testing acc'로 지정
    plt.legend()
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')     # hist 딕셔너리에서 학습 손실도(train_loss)를 추출하여 그래프로 그리는데 라벨을 'Training loss'로 지정
    plt.plot(hist['test_loss'], label='Testing loss')       # hist 딕셔너리에서 검증 손실도(test_loss)를 추출하여 그래프로 그리는데 라벨을 'Testing loss'로 지정
    plt.legend()

# 컨볼루션(Convolution) 연산을 시각화하는 함수 plot_convolution을 정의하는데 특정 커널을 사용하여 이미지에 적용한 결과를 보여줌 
def plot_convolution(t,title=''):
    with torch.no_grad():
        c = nn.Conv2d(kernel_size=(3,3),out_channels=1,in_channels=1) #커널 사이즈 3,3 그레이 스케일로 감
        c.weight.copy_(t)
        fig, ax = plt.subplots(2,6,figsize=(8,3))
        fig.suptitle(title,fontsize=16)
        for i in range(5):
            im = data_train[i][0]
            ax[0][i].imshow(im[0])
            ax[1][i].imshow(c(im.unsqueeze(0))[0][0])
            ax[0][i].axis('off')
            ax[1][i].axis('off')
        ax[0,5].imshow(t)
        ax[0,5].axis('off')
        ax[1,5].axis('off')
        #plt.tight_layout()
        plt.show()

# 주어진 데이터셋에서 이미지를 선택하여 시각화하는 Python 함수 display_dataset을 정의하는데 이미지 데이터셋, 표시할 이미지의 수, 그리고 선택적으로 클래스 레이블을 포함할 수 있음
def display_dataset(dataset, n=10,classes=None):
    fig,ax = plt.subplots(1,n,figsize=(15,3))
    mn = min([dataset[i][0].min() for i in range(n)])
    mx = max([dataset[i][0].max() for i in range(n)])
    for i in range(n):
        ax[i].imshow(np.transpose((dataset[i][0]-mn)/(mx-mn),(1,2,0)))
        ax[i].axis('off')
        if classes:
            ax[i].set_title(classes[dataset[i][1]])


# 주어진 파일 이름(fn)에 해당하는 이미지 파일을 검사하여 파일이 유효한 이미지인지 확인하는 Python 함수 check_image를 정의
def check_image(fn):
    try:
        im = Image.open(fn)
        im.verify()
        return True
    except:
        return False

# 지정된 경로에 있는 모든 이미지 파일을 검사하고 손상된 이미지 파일을 찾아 삭제하는 Python 함수 check_image_dir를 정의
def check_image_dir(path):
    for fn in glob.glob(path):
        if not check_image(fn):
            print("Corrupt image: {}".format(fn))
            os.remove(fn)

# PyTorch의 torchvision 라이브러리를 사용하여 이미지 변환을 위한 일반적인 변환 조합을 설정하는 함수 common_transform을 정의
# 전이 학습(Transfer Learning)이나 컴퓨터 비전 모델에서 이미지를 전처리할 때 매우 유용한데 이 변환을 사용하면 학습 데이터와 테스트 데이터를 모델이 기대하는 형식으로 일관되게 처리할 수 있음

def common_transform():
    std_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(), 
            std_normalize])
    return trans

# 사용하지 않음
def load_cats_dogs_dataset():
    if not os.path.exists('data/PetImages'):
        with zipfile.ZipFile('data/kagglecatsanddogs_5340.zip', 'r') as zip_ref:
            zip_ref.extractall('data')

    check_image_dir('data/PetImages/Cat/*.jpg')
    check_image_dir('data/PetImages/Dog/*.jpg')

    dataset = torchvision.datasets.ImageFolder('data/PetImages',transform=common_transform())
    trainset, testset = torch.utils.data.random_split(dataset,[20000,len(dataset)-20000])
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32)
    testloader = torch.utils.data.DataLoader(trainset,batch_size=32)
    return dataset, trainloader, testloader