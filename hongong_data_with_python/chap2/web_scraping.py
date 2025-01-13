import sys, os
# 현재 파일의 부모 디렉터리를 sys.path에 추가하여 상위 폴더의 파일들을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 

import pandas as pd
from bs4 import BeautifulSoup
import requests

def get_page_cnt(isbn):
    # isbn으로 검색
    url = 'https://www.yes24.com/Product/Search?domain=ALL&query={}'
    response_isbn_web = requests.get(url.format(isbn))
    # 도서 상세페이지 구하기
    soup_isbn_web = BeautifulSoup(response_isbn_web.text, 'html.parser')
    book_juso_info = soup_isbn_web.find('a', attrs={'class':'gd_name'})
    if book_juso_info == None:
        print('검색 실패. 책을 발견하지 못함')
        return
    
    # 도서 상세페이지 접속 (나미 어쩌구는 전체 url이 들어가 있음)
    if book_juso_info['href'].startswith("https://"):
        url2 = book_juso_info['href']
    else:
        url2 = 'https://www.yes24.com'+book_juso_info['href']

    response_book_web = requests.get(url2)
    soup_book_web = BeautifulSoup(response_book_web.text, 'html.parser')
    book_info = soup_book_web.find('div', attrs={'id':'infoset_specific'})
    # 책의 쪽수, 무게, 크기를 추출 (나미야 잡화점의 기적은 절판된거 같다. 중고밖에 없음)

    try:
        book_tr_list = book_info.find_all('tr')
    except:
        # 책이 판매하고 있지 않음
        print(book_juso_info.text, '는 현재 판매를 하지 않는 것 으로 추정')
        return None
    
    for tr in book_tr_list:
        if tr.find('th').get_text() == '쪽수, 무게, 크기':
            page_td = tr.find('td').get_text()
            return page_td.split()[0]
        
    print("책의 정보를 발견 못함")
    return 

def get_page_from_df(df):
    df = df['ISBN']
    return  get_page_cnt(df)

if __name__=='__main__':
    df = pd.read_json('data\\20s_best_book.json')
    books = df[['순위', '서명', '저자','출판사', '출판년도','ISBN']]

    books = books.head(10)
    page_cnts = books.apply(get_page_from_df, axis=1)
    print(page_cnts)