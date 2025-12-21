import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

#пасхалку нашёл, но объяснить её...

# Загружаем данные
df = pd.read_csv('dataset_simple.csv')

#X = torch.Tensor(df.iloc[:, 0:2].values)  # age и income
#y = torch.Tensor(df.iloc[:, 2].values).reshape(-1, 1)  # will_buy

# Без нормализации данных ответ был постоянно 69%

age_mean, age_std = df.iloc[:, 0].mean(), df.iloc[:, 0].std()
income_mean, income_std = df.iloc[:, 1].mean(), df.iloc[:, 1].std()

# Нормализуем признаки
df['age_norm'] = (df.iloc[:, 0] - age_mean) / age_std
df['income_norm'] = (df.iloc[:, 1] - income_mean) / income_std

# Классификация: предсказание will_buy по age и income
X = torch.Tensor(df[['age_norm', 'income_norm']].values)  # нормализованные age и income
y = torch.Tensor(df.iloc[:, 2].values).reshape(-1, 1)  # will_buy

# Создаем нейронную сеть для бинарной классификации
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid()  # Бинарная классификация
        )
    
    def forward(self, x):
        pred = self.layers(X)
        return pred

# Параметры сети
inputSize = X.shape[1]  # 2 признака: age и income
hiddenSize = 3
outputSize = 1  # один выход: вероятность покупки

net = NNet(inputSize, hiddenSize, outputSize)

# Функция потерь и оптимизатор
lossFn = nn.BCELoss()  # Бинарная классификация
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Обучение (Повышаем количество шагов т.к не успевало обучиться за 100)
epochs = 300
for i in range(epochs):
    pred = net.forward(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

# Оценка точности
with torch.no_grad():
    pred = net.forward(X)
    pred_1 = (pred >= 0.5).float()
    accuracy = (pred_1 == y).float().mean()

print(f'\nТочность модели: {accuracy.item()*100:.2f}%')

pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))

# Считаем количество ошибочно классифицированных примеров (Выдаёт чушь)
#err = sum(abs(y-pred))/2 
#print(err)
    