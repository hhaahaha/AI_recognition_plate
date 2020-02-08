###=========================================
# import trained model for transfer learning
# refer - https://tutorials.pytorch.kr/beginner/saving_loading_models.html#id8
#PATH = './mnist_resnet.pth'
#PATH = './SVHN_resnet.pth'
# PATH1 = './SVHN_resnet_model.pth'
# torch.save(model_ft.state_dict(), PATH)
# torch.save(model_ft, PATH1)
###========================================

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # 대화형 모드

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import visdom

import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

#학습시킨 모델 불러오기
resnet18 = models.resnet18(pretrained=True)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)

PATH = './SVHN_resnet.pth'
PATH1 = './SVHN_resnet_model.pth'

resnet18.load_state_dict(torch.load(PATH))
resnet18.eval()

learned_model = torch.load(PATH1)
learned_model.eval()

block = BasicBlock
print(block)
layers = [2, 2, 2, 2]


#모델 커스텀하기
class SVHNResNet(nn.Module):
    def __init__(self):
        super(SVHNResNet, self).__init__() # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        self.layer0 = nn.Sequential(*list(learned_model.children())[0:-1])
        self.layer1 = nn.Sequential(
            nn.Linear(512, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, num_category),
            nn.ReLU()
        )
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)

    def forward(self,x):

        out = self.layer0(x)
        # print('layer0 fin')
        # view 함수는 n차원 텐서를 1차원 텐서로 바꿔준다. 이때, (64,28,28,1) 은 각각 (batchsize , h, w, c) 이다. view 함수로 0번째 인수를 batch size로 잡아주는 이유는 이미지끼리 섞이면 안 되기 때문이다.
        out = out.view(batch_size_train, -1)
        # print('tensor reshape')
        out = self.layer1(out)
        # print('layer1 fin')
        # print(out.shape)
        return out


# print('걍 만든 모델',learned_model)
# trans_model = SVHNResNet()
# trans_model.load_state_dict(torch.load(PATH))
#
# trans_model.eval()

#
# print()
# print('파라미터 덮어씌운 모델 \n',trans_model)


#traning / val data 불러오기
# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화
data_transforms = {
    'train': transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0),
        transforms.RandomVerticalFlip(p=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'Nogada_Char'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=4, drop_last = True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

#hyperparameters

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
num_epoch = 10
num_category = len(image_datasets['train'].classes)
class_names = image_datasets['train'].classes

# 모델 학습시키기
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.ioff()
    plt.show()
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.


### 모델 train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                # labels.ToTensor()
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # print('outputs',outputs.shape)
                    # print('label',labels.shape)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

#visualizing predict
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            # print('input shape',inputs.shape)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


#새로운 모델 생성

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transfer_model = SVHNResNet().to(device)

#모델의 layer0은 학습되지 않게 기울기 계산을 꺼둔다.
for params in transfer_model.layer0.parameters():
    params.require_grad = False

#모델의 layer1은 학습되도록 기울기 계산을 켜둔다.
for params in transfer_model.layer1.parameters():
    params.require_grad = True

# 모델을 한번 확인합니다
print('모델 체크한다')
for m in transfer_model.children():
    print(m)



model_ft = transfer_model.to(device)

criterion = nn.CrossEntropyLoss()

# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 7 에폭마다 0.1씩 학습율 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#train and eval
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

visualize_model(model_ft)


plt.ioff()
plt.show()

print('\n after train model \n', model_ft)