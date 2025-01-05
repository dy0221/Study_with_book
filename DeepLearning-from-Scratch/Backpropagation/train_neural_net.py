import sys, os
# 내경로가 현재 폴더가 아닌 상위 폴더로 바꿈 # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from Backpropagation.two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


# 서브플롯 생성
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1행 2열 서브플롯

# 첫 번째 그래프: Training Loss
axs[0].plot(range(len(train_loss_list)), train_loss_list)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss over Iterations')
axs[0].grid(True)

# 두 번째 그래프: Training & Test Accuracy
x = np.arange(len(train_acc_list))
axs[1].plot(x, train_acc_list, label='train acc', marker='o')
axs[1].plot(x, test_acc_list, label='test acc', linestyle='--', marker='s')
axs[1].set_xlabel("epochs")
axs[1].set_ylabel("accuracy")
axs[1].set_ylim(0, 1.0)
axs[1].legend(loc='lower right')
axs[1].set_title('Training and Test Accuracy')

plt.tight_layout()  # 그래프 간 간격 조정
plt.show()