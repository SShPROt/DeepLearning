# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 21:20:08 2025

@author: vladk
"""
import torch
import random

x = torch.tensor(random.randint(1, 3), dtype=torch.int32)
print("1. Целочисленный тензор: ", x)

x = x.float()
print("2. Тензор с плавабщей точкой: ", x)

n = 2 #Вариант 11
x.requires_grad_(True)
x = x ** n
print("3.1. Тензор в степени: ", x)

multiplier = random.uniform(1, 3)
x = x * multiplier
print("3.2. Тензор умножили: ", x)

x = torch.exp(x)
print("3.3. Взяли экспоненты: ", x)

x.backward()
print("4. Производная: ", x)

