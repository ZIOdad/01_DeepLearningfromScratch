#1.6 matplotlib
import numpy as np
import matplotlib.pyplot as plt

#데이터 생성
x = np.arange(0, 6, 0.1) #0에서 6까지 0.1간격으로 생성
y = np.sin(x)

#그래프 그리기
plt.plot(x,y)
plt.show()

#1.6.2 pylot 기능
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin") #y1 label명
plt.plot(x, y2, linestyle="--", label="cos") #y2 선모양 및 label명
plt.xlabel("x") #x축 이름
plt.ylabel("y") #x축 이름
plt.title('sin & cos') #그래프 이름
plt.legend() #범례표시
plt.show()

#1.6.3 이미지불러오기
from matplotlib.image import imread

img = imread('D:\\1_Python_code\\01_Deep Learning from scratch\\Yosemite.jpg')

plt.imshow(img)
plt.show()
