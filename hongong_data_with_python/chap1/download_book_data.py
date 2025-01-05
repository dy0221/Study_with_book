"""
 $ pip install gdown
"""
import sys, os
# 현재 파일의 부모 디렉터리를 sys.path에 추가하여 상위 폴더의 파일들을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 

import gdown

gdown.download('https://bit.ly/3eecMKZ',
               'data\\남산도서관 장서 대출목록 (2021년 04월).csv', quiet=False)