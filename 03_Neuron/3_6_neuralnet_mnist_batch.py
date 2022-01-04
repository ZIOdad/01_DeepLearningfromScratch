#손글씨 숫자 인식 2.신경망(추론) 구축
#입력층 뉴런 784개(28X28 이미지파일 픽셀수, 0~255)
#출력층 뉴런 10개(분류항목수, 0~9)
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle #파이썬의 모든 객체(Object)에 대하여 그대로 저장할 수 있는 모듈
              #binary 파일로 저장하기 때문에 'rb', 'wb' 형식을 사용
from dataset.mnist import load_mnist
from common.functions import sigmoid
from common.functions import softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
batch_size = 100 #배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p= np.argmax(y_batch, axis=1) # 확률이 가장 높은 원소의 인덱스 *axis->행(row)과 열(colunms)를 구분
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) #p == t[i:i+batch_size] 논리연산을 통해 Ture(1)를 모두 더함

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
