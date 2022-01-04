import numpy as np
#출력층 설계(분류와 회귀)
#항등함수(회귀)
def identity_function(x):
    return x

#소프트맥스함수(분류)
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)

y = exp_a / np.sum(exp_a)
print(y)

a = np.array([1010, 1000, 990])
def softmax(x):
    c=np.max(x)
    exp_x = np.exp(x-c)
    y = exp_x / np.sum(exp_x)
    return y
print(softmax(a))
print(np.sum(softmax(a))) #소프트맥스함수는 0~1 사이 실수를 출력하고 출력값의 총합은 1임.
#소프트맥스함수의 성질을 이용하여 '확률'로 해석가능 -> 확률적인 판단을 통해 분류(classification)가 가능해짐.
#지수함수 = 단조증가함수이므로 원소들간의 대소관계가 변하지 않음.
#지수함수 계산비용을 줄이기 위해 추론단계에서는 소프트맥스함수 생략함.
