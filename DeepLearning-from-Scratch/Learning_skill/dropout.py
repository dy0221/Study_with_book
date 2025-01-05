import numpy as np

class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        # 훈련할때
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
        # 테스트 할때 출력층의 개수 차이로 인한 오차를 줄이기 위해 사용
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask