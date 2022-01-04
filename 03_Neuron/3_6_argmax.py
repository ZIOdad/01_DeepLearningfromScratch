import numpy as np

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=1)
print(x.shape)
print(y.shape)
print(y) #[1 2 1 0] columns 방향으로 가장 큰 인수를 가진 인덱스

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=0)
print(x.shape)
print(y.shape)
print(y) #[3 0 1] row 방향으로 가장 큰 인수를 가진 인덱스
