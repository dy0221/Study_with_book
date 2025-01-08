"""
안전한 torch.load 사용 권고(weights_only=True)
현재는 weights_only=False가 기본값인데 보안상의 취약점이 있을 수 있다.
따라서 신뢰할수 없는 파일이나 데이터 로드의 경우 weights_only=True를 명시적으로 설정하라고 한다.

FutureWarning: You are using `torch.load` with `weights_only=False` 
(the current default value), which uses the default pickle module implicitly.
It is possible to construct malicious pickle data which will execute arbitrary 
code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details).
In a future release, the default value for `weights_only` will be flipped to `True`. 
This limits the functions that could be executed during unpickling.
Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`.
We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.

"""
import sys, os
# 현재 파일의 부모 디렉터리를 sys.path에 추가하여 상위 폴더의 파일들을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
import torch

points = torch.load('data/tensor/ourpoints.t',weights_only=True)

print(points)