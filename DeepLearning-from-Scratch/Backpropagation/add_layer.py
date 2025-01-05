class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1 # 그대로 전달
        dy = dout * 1

        return dx, dy