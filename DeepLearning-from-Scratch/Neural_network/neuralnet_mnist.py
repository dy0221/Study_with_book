import sys, os
# 내경로가 현재 폴더가 아닌 상위 폴더로 바꿈 # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
import numpy as np
from dataset.mnist import load_mnist
from Neural_network.sigmoid import sigmoid
from Neural_network.softmax import softmax
import pickle
import time

class NEURALNET_MNIST:

    def __init__(self):
        self.network = None
        self.W1 = self.W2 = self.W3 = None
        self.b1 = self.b2 = self.b3 = None
        self.init_network()

    def get_data(self):
        (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=True, flatten=True, one_hot_label=False)
        return x_test, t_test
    
    def init_network(self):
        with open("Neural_network\\sample_weight.pkl", 'rb') as f:
            self.network = pickle.load(f)

    def predict(self, x):
        self.W1, self.W2, self.W3 = self.network['W1'], self.network['W2'], self.network['W3']
        self.b1, self.b2, self.b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, self.W1) + self.b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.W2) + self.b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, self.W3) + self.b3
        y = softmax(a3)

        return y
    
if __name__=='__main__':

    neuralnet_mnist = NEURALNET_MNIST()
    x, t = neuralnet_mnist.get_data()

    batch_size = 100
    accuracy_cnt = 0
    accuracy_cnt2 = 0
    print(x.shape)
    cnt = 0
    time_start = time.time()
    for i in range(len(x)):
        y = neuralnet_mnist.predict(x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
        cnt += 1
    during_time = time.time() - time_start
    print("cnt : ",cnt)
    print("time without batch : ", during_time)
    print("Accuracy:" ,str(float(accuracy_cnt) / len(x)))

    cnt = 0
    time_start = time.time()
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = neuralnet_mnist.predict(x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt2 += np.sum(p == t[i:i+batch_size])
        cnt += 1
    during_time = time.time() - time_start

    print("cnt : ",cnt)
    print("time with batch :    ", during_time)
    print("Accuracy with batch:" ,str(float(accuracy_cnt) / len(x)))