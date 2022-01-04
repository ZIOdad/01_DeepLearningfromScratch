#신경망(Neuron) : 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습
# 퍼셉트론과의 가장 큰 차이점!
# 활성화함수(activation fuction)의 도입 -> 1 or 0, on/off
# 이러한 활성화함수를 통해 퍼셉트론(수동)에서 신경망(자동)으로 진화됨
# 다층 구조의 신경망의 이점을 얻기 위해서는 활성화함수로 비선형함수를 사용해야함

#3.2.2 계단함수
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

#넘파이 배열을 인수로 사용할 수 있도록 수정
def step_function(x):
    y = x > 0
    return y.astype(np.int)

import numpy as np
x = np.array([-1.0, 1.0, 2.0])
print(x)
y = x > 0
print(y)
y = y.astype(np.int)
print(y)

#계단함수 그래프
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y, label="step", linestyle="--")
plt.ylim(-0.1, 1.1) #y축 범위

#3.2.4 시그모이드함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#시그모이드함수 그래프
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
plt.plot(x, y1, label="sigmod")
plt.ylim(-0.1, 1.1) #y축 범위
plt.show()

#3.2.7 ReLU 함수
def relu(x):
    return np.maximum(0, x)
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y, label="relu")
plt.ylim(-0.1, 5.1) #y축 범위
plt.show()
