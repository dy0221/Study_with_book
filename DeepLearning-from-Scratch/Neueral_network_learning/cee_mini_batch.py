import sys, os
# 내경로가 현재 폴더가 아닌 상위 폴더로 바꿈 # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
import numpy as np
from dataset.mnist import load_mnist
from Neural_network.neuralnet_mnist import NEURALNET_MNIST
from Neueral_network_learning.cross_entropy_error import cross_entropy_error
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
# print(x_batch.shape)
# print(t_batch.shape)

NM = NEURALNET_MNIST()
y = []
for i in range(batch_size):
    y.append(NM.predict(x_batch[i]))

y = np.array(y)
# print(y.shape)
print(f"Cross entropy error: {cross_entropy_error(y, t_batch)}")
    
    
