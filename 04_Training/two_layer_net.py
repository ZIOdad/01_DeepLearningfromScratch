#학습 알고리즘
"""
신경망 학습 순서
*1단계 - 미니배치
training 데이터 중 무작위로 선별한 데이터를 미니배치라고 함.
이 미니배치의 손실함수 값을 줄이는 것이 목표임.
*2단계 - 기울기 산출
미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기(dL/dW)를 구함.
기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시함.
*3단계 - 가중치 매개변수 갱신
가중치 매개변수를 기울기 방향으로 갱신함.
*4단계 - 반복
1~3단계를 반복하여 신경망 내 가중치와 편향을 학습함.
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x: 입력데이터, t:정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x: 입력데이터, t:정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    #오차역전파법
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

#class내 인스턴스(딕셔너리) 변수 shape확인
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape) #(784, 100)
print(net.params['b1'].shape) #(100,)
print(net.params['W2'].shape) #(100, 10)
print(net.params['b2'].shape) #(10,)

x = np.random.rand(100, 784) #더미 입력 데이터
t = np.random.rand(100, 10) #더미 정답 레이블
y = net.predict(x)

#grads = net.numerical_gradient(x, t)
grads = net.gradient(x, t)

print(grads['W1'].shape) #(784, 100)
print(grads['b1'].shape) #(100,)
print(grads['W2'].shape) #(100, 10)
print(grads['b2'].shape) #(10,)
