"""
DataFrame 요소 접근
"""
import sys, os
# 현재 파일의 부모 디렉터리를 sys.path에 추가하여 상위 폴더의 파일들을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 

import pandas as pd

df = pd.read_csv('data\\남산도서관 장서 대출목록 (2021년 04월).csv', encoding='EUC-KR', 
                 dtype={'ISBN': str, '주제분류번호': str})

print(df.columns)
print(df.index)

type_data = df['세트 ISBN'].iloc[109]
type2_data = df['세트 ISBN'].iloc[105]
print(type(type_data))
print(type(type2_data))

