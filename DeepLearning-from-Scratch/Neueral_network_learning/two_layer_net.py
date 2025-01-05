import sys, os
# 내경로가 현재 폴더가 아닌 상위 폴더로 바꿈 # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
import numpy as np
from Neueral_network_learning.numetical_gradient import numerical_gradient
from Neueral_network_learning.cross_entropy_error import  cross_entropy_error
from Neural_network.sigmoid import sigmoid
from Neural_network.softmax import softmax
import time

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
       

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2'] 

        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        acurracy = np.sum(y == t) / float(x.shape[0])
        return acurracy
    
    def numercial_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
if __name__=='__main__':
    net = TwoLayerNet(input_size = 784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)

    x = np.random.rand(100, 784)
    t = net.predict(x)
    
    start_time = time.time()
    grads = net.numercial_gradient(x, t)
    end_time = time.time()

    print(f"Gradient calculation took {end_time - start_time:.2f} seconds")
    print(grads['W1'].shape)
    print(grads['b1'].shape)
    print(grads['W2'].shape)
    print(grads['b2'].shape)

