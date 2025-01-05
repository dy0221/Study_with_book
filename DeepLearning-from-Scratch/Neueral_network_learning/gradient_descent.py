import numpy as np
from numetical_gradient import numerical_gradient

# f(x0,x1) = x0^2 + x1^2
def function_2(x):
    return x[0]**2 + x[1]**2

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    
    return x


if __name__=='__main__':
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x, 0.1, 100))