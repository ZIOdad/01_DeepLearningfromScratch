import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=True)

#각 데이터의 형상 출력
print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000,10)
print(x_test.shape) #(10000, 784)
print(t_test.shape) #(10000,10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
print(x_batch.shape)
t_batch = t_train[batch_mask]
print(t_batch.shape)

"""#배치용 교차엔트로피오차 -> predicted y가 없으므로 에러 발생
def cross_entropy_error(y, t):
    if y.ndim == 1: #1차원인 경우, 즉 데이터하나당 오차를 구할 때는 데이터 형상을 바꿔줌
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    #return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size  *one_hot_encoding 형태일 경우
print(cross_entropy_error(np.array(x_batch), np.array(t_batch)))
"""
