#수치미분
def numerical_diff(f, x):
   h = 1e-4 #0.0001
   return (f(x+h) - f(x-h)) / (2*h)  #중심차분(2nd order error)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

def function_2(x):
    return x[0]**2 + x[1]**2 #return np.sum(x**2)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) #x와 형상(shape)이 같은 0을 원소로 가진 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # x값 복원

    return grad

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 10)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
