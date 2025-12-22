"""
Created on Tue Apr  6 21:05:20 2021

@author: AM4
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
                        transforms.Resize(68),
                        transforms.CenterCrop(64),
                        transforms.ToTensor()])

# ImageFolder указывает что датасет загружается из каталога.
train_dataset = torchvision.datasets.ImageFolder(root='./MyDataset/train',
                                                 transform=data_transforms)


# И отдельно создадим тестовый набор.
test_dataset = torchvision.datasets.ImageFolder(root='./MyDataset/test',
                                             transform=data_transforms)


# посмотрим какие классы содержатся в наборе
train_dataset.classes

# сохраним названия этих классов
class_names = train_dataset.classes

# Список изображений можно получить следующим образом:
train_set = train_dataset.samples
print(train_set[1]) #  каждая строка списка содержит путь к изображению и метку класса

# посмотрим на размер нашего набора данных
print(len(train_set))

batch_size = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True,  num_workers=0)


test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=0) 


inputs, classes = next(iter(train_loader))
inputs.shape 

# построим на картинке
img = torchvision.utils.make_grid(inputs, nrow = 5)
img = img.numpy().transpose((1, 2, 0)) 
plt.imshow(img)


class CnNet(nn.Module):
    def __init__(self, num_classes=12):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
        # первый сверточный слой с ReLU активацией и maxpooling-ом
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        # второй сверточный слой 
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        # третий сверточный слой 
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.fc = nn.Linear(2*2*64, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1) # флаттеринг
        out = self.fc(out)
        return out



# Количество классов
num_classes = 3

# создаем экземпляр сети
net = CnNet(num_classes).to(device)

# Задаем функцию потерь и алгоритм оптимизации
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# создаем цикл обучения и замеряем время его выполнения
import time
t = time.time()

# Зададим количество эпох обучения (каждая эпоха прогоняет обучающий набор 1 раз)
num_epochs = 50
save_loss = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # прямой проход
        outputs = net(images)
        
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)
        
        # Обратный проход (вычисляем градиенты)
        optimizer.zero_grad()
        loss.backward()
        
        # делаем шаг оптимизации весов
        optimizer.step()
        
        # сохраняем loss
        save_loss.append(loss.item())
        # выводим немного диагностической информации
        if i%100==0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print(time.time() - t)

# Посмотрим как уменьшался loss в процессе обучения
plt.figure()
plt.plot(save_loss)

# Посчитаем точность нашей модели: количество правильно классифицированных картинок
# поделенное на общее количество тестовых примеров
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()


print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

# Нашу модель можно сохранить в файл для дальнейшего использования
torch.save(net.state_dict(), 'CnNet.ckpt')

data_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225] )
    ])


# Пересоздадим датасеты с учетом новых размеров и нормировки яркости
train_dataset = torchvision.datasets.ImageFolder(root='./MyDataset/train',
                                                 transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root='./MyDataset/test',
                                             transform=data_transforms)


batch_size = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True,  num_workers=0)


test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=0) 


net = torchvision.models.alexnet(pretrained=True)

# можно посмотреть структуру этой сети
print(net)

for param in net.parameters():
    param.requires_grad = False

num_classes = 3

new_classifier = net.classifier[:-1]
new_classifier.add_module('fc',nn.Linear(4096,num_classes))
net.classifier = new_classifier

net = net.to(device)

# проверим эффективность новой сети
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')


# Перейдем к обучению.
# Зададим количество эпох обучения, функционал потерь и оптимизатор.
num_epochs = 2
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# создаем цикл обучения и замеряем время его выполнения
t = time.time()
save_loss = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # прямой проход
        outputs = net(images)
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)
         # Обратный проход (вычисляем градиенты)
        optimizer.zero_grad()
        loss.backward()
        # делаем шаг оптимизации весов
        optimizer.step()
        save_loss.append(loss.item())
        # выводим немного диагностической информации
        if i%100==0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print(time.time() - t)

# Посмотрим как уменьшался loss в процессе обучения
plt.figure()
plt.plot(save_loss)

# Еще раз посчитаем точность нашей модели
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
# уже лучше


# Реализуем отображение картинок и их класса, предсказанного сетью
inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

for i,j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[j])
    plt.pause(2)