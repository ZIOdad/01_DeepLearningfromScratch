import numpy as np

#항등함수(회귀)
def identity_function(x):
    return x

#계단함수
def step_function(x):
    return np.array(x > 0, dtype=np.int)

#시그모이드함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#시그모이드구배
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

#ReLU함수
def relu(x):
    return np.maximum(0, x)

#소프트맥스함수
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

#평균제곱오차(Mean squared error, MSE)
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

#교차엔트로피오차(Cross entropy error, CEE)
def cross_entropy_error(y, t):
    if y.ndim == 1: #1차원인 경우, 즉 데이터하나당 오차를 구할 때는 데이터 형상을 바꿔줌
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size  #*one_hot_encoding 형태일 경우
