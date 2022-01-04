#다차원 배열 계산
import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
print(B.shape[0])

#2X2 * 2X2 = 2X2
A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))

#2X3 * 3X2 = 2X2
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)
print(np.dot(A, B))

#3X2 * 2X1 = 3X1
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)
B = np.array([7, 8])
print(B.shape)
print(np.dot(A, B))

#신경망에서의 행렬 곱
X = np.array([1, 2]) #독립변수
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]]) #가중치(weight)
print(W.shape)
Y = np.dot(X, W) #종속변수
print(Y)
