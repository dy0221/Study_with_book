import sys,os
import requests
# 현재 파일의 부모 디렉터리를 sys.path에 추가하여 상위 폴더의 파일들을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
# GitHub의 raw URL 주소를 사용해야 한다.(api 주소)
github_address = 'https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch4/volumetric-dicom/2-LUNG%203.0%20%20B70f-04083/'

output_dir = "data/dicom/"

# 파일 이름 자동 생성
file_names = [f"0000{str(i).zfill(2)}.dcm" for i in range(0, 99)]
print(file_names)

# 파일 다운로드
for file_name in file_names:
    file_url = github_address + file_name
    local_path = os.path.join(output_dir, file_name)

    response = requests.get(file_url)
    
    # 200: 요청 성공 (파일 다운로드 가능).
    # 404: 요청한 파일이 존재하지 않음 (Not Found).
    # 500: 서버 에러 (Server Error).

    if response.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download {file_url}. Status code: {response.status_code}")
