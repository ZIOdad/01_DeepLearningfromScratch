import numpy as np
#1.5.2 넘파이 배열 생성
x = np.array([1,2,3,4])
print(x)
print(type(x))

#1.5.3 넘파이 산술 연산 (행렬 원소끼리의 연산!)
y = np.array([2,3,4,5])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x/2.0)
z=x/2.0
print(z.dtype)

#1.5.4 넘파이 N차원 배열
A = np.array([[1,2],[3,4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3,0],[0,6]])
print(A+B)
print(A*B)
print(A*10)

#1.5.5 브로드캐스트(broadcast)
A = np.array([[1,2],[3,4]])
B = np.array([10,20])
print(A*B)
print("\n")

#1.5.6 원소 접근
X = np.array([[51,55],[14,19],[0,4]])
print(X)
print(X[0])
print(X[0][1])
print("\n")

for row in X:
    print(row)

X = X.flatten()
print(X)
print(X[np.array([0,2,4])])
