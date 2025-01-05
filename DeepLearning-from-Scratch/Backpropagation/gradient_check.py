# coding: utf-8
import sys, os
# 내경로가 현재 폴더가 아닌 상위 폴더로 바꿈 # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 

import numpy as np
from dataset.mnist import load_mnist
from Backpropagation.two_layer_net import TwoLayerNet

    # 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

print('수치 미분 학습 (순정파)')
grad_numerical = network.numerical_gradient(x_batch, t_batch)
print('역전파 학습')
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    print('key : ', key)
    print('순정파 :',grad_numerical[key].shape)
    print('역전파 :',grad_backprop[key].shape)
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print('기울기 오차' + ":" + str(diff))