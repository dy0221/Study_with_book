import sys, os
# 현재 파일의 부모 디렉터리를 sys.path에 추가하여 상위 폴더의 파일들을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
import torch

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

torch.save(points, 'data/tensor/ourpoints.t')