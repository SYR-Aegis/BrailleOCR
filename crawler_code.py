import requests
from bs4 import BeautifulSoup
import re

url='https://www.msn.com/ko-kr/news/living/%EC%86%90%EB%B0%9C-%EC%A0%80%EB%A6%BC-%ED%98%88%EC%95%A1%EC%88%9C%ED%99%98%EC%9D%B4-%EC%95%84%EB%8B%8C-%EC%8B%A0%EA%B2%BD-%EB%AC%B8%EC%A0%9C/ar-BB1cVbuE?ocid=msedgntp'
resp = requests.get(url)

if(resp.status_code==200): #정상 처리
    soup = BeautifulSoup(resp.text, 'html.parser') #parsing위해 BeautifulSoup 객체 생성
    body = soup.find_all('body')

    contents='' #태그 안의 value값 추출해 저장하는 문자열 변수
    for i in body:
        contents += i.get_text()


    if not new_contents[0].isalpha():
        new_contents[0] = ''

    #한 문장 끝나면 줄바꿈
    for i in range(len(new_contents)):
        if(new_contents[i] == '.'):
            if(new_contents[i+1] !='\n'):
                new_contents[i + 1] == '\n'

    new_contents = re.findall('[가-힣\n ]', contents)  #순수 한글,줄바꿈,띄어쓰기만 뽑아냄 #각 문자가 리스트에 저장

    #\n가 한번 나오고 다시 한글 나올때까지 모든 특수 문자 제거
    for i in range(len(new_contents)):
        if(new_contents[i] == '\n'):
            for k in range(len(new_contents)-i-1):
                if (new_contents[i + k + 1].isalpha()):
                    break
                else:
                    new_contents[i + k + 1] = ""

    #띄어쓰기 중복 제거
    for i in range(len(new_contents) - 1):
        if (new_contents[i] == ' '):
            if (new_contents[i + 1] == ' '):
                new_contents[i] = ""

    #마지막 줄바꿈 제거
    for i in range(len(new_contents)-1,0,-1):
        if(new_contents[i]=='\n'):
            new_contents[i]=''
        if(new_contents[i].isalpha()):
            break

    new_contents[len(new_contents)].append('\n')


    #텍스트 파일에 저장
    f = open('crawl.txt', 'a')
    f.write("".join(new_contents)) #리스트에 저장된 각 문자들을 문자열로 결합
    f.close()
