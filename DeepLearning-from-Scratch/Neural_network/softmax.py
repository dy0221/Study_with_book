import numpy as np
# 이거 사용하니 오버 플로 났던거 같다.
# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a-c)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a

#     return y

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

if __name__=='__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)