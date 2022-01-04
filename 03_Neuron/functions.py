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

#ReLU함수
def relu(x):
    return np.maximum(0, x)

#소프트맥스함수
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
