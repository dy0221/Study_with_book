import numpy as np
import matplotlib.pyplot as plt

from AdaGrad_optimizer import AdaGrad
from Adam_optimizer import Adam
from Momentum_optimizer import Momentum
from SGD_optimizer import SGD

# Rosenbrock 함수와 그라디언트 정의
def f_rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f_rosenbrock(x):
    df_dx0 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    df_dx1 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx0, df_dx1])

# 옵티마이저 업데이트 함수
def optimize(optimizer, init_x=np.array([2.0, 2.0]), num_iterations=2000):
    x = np.copy(init_x)
    history = [x.copy()]
    points_500 = []  # 500번마다의 위치 저장
    
    for i in range(num_iterations):
        grads = grad_f_rosenbrock(x)
        
        optimizer.update({'x': x}, {'x': grads})
        history.append(x.copy())
        
        if i % 500 == 0:  # 500번마다 진행 상황 출력
            points_500.append(x.copy())  # 500번마다의 위치 기록
            print(f"optimizer {optimizer.__class__.__name__}, Iteration {i}, x: {x}, f(x): {f_rosenbrock(x)}")
            if optimizer.__class__.__name__ == 'AdaGrad':
                print(optimizer.h)

    return history, points_500

# 각 옵티마이저별 학습률 설정
optimizers = [
    AdaGrad(lr=0.01),  
    Adam(lr=0.001),  
    Momentum(lr=0.001),
    SGD(lr=0.001)
]

labels = ['AdaGrad', 'Adam', 'Momentum', 'SGD']
colors = ['b', 'g', 'r', 'c']

# 2x2 서브플롯 만들기
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()  # 2x2 배열을 1D 배열로 변환

# 옵티마이저마다 결과 시각화
for optimizer, label, color, ax in zip(optimizers, labels, colors, axes):
    history, points_500 = optimize(optimizer, init_x=np.array([2.0, 2.0]))  # 초기값을 더 작은 값으로 설정
    history_x = [h[0] for h in history]
    history_y = [h[1] for h in history]
    
    # 경로를 좀 더 선명하게 그리기
    ax.plot(history_x, history_y, label=label, color=color, alpha=0.7)  # alpha로 투명도를 조절
    
    # 시작점과 끝점을 명확히 표시
    ax.scatter(history_x[0], history_y[0], color=color, marker='o', s=50, label=f'{label} Start')
    ax.scatter(history_x[-1], history_y[-1], color=color, marker='x', s=100, label=f'{label} End')

    # 500번마다의 위치 표시
    for point in points_500:
        ax.scatter(point[0], point[1], color=color, marker='D', s=80, alpha=0.5, edgecolor='black')

    # Rosenbrock 함수의 등고선 추가
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 100 * (Y - X**2)**2 + (1 - X)**2
    ax.contour(X, Y, Z, levels=np.logspace(0, 5, 20), cmap='gray', linestyles='dotted')

    # 그래프에 레이블 추가
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'{label} Optimizer')
    ax.grid(True)
    ax.legend()

# 서브플롯 간 간격 조정
plt.tight_layout()
plt.show()
