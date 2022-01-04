#손글씨 숫자 인식 1.준비단계

#mnist 데이터 다운로드
import sys, os
sys.path.append(os.pardir) #mnist가 상위 디렉토리에 있기때문에 추가
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=False, flatten=True, one_hot_label=False)

#각 데이터의 형상 출력
print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000,)
print(x_test.shape) #(10000, 784)
print(t_test.shape) #(10000,)

#mnist 데이터 이미지 출력
import numpy as np
from PIL import Image #PIL=사진 관련 라이브러리

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) #넘파이로 저장된 이미지 데이터를 PIL용 데이터로 변환
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 원래 이미지의 모양으로 변형
print(img.shape)  # (28, 28)

img_show(img)
