import requests
from bs4 import BeautifulSoup
import re

url='https://namu.wiki/w/%EC%84%B1%ED%83%84%EC%A0%88'
resp = requests.get(url)

if(resp.status_code==200): #정상 처리
    soup = BeautifulSoup(resp.text, 'html.parser') #parsing위해 BeautifulSoup 객체 생성

    if 'namu.wiki' in url:
        body = soup.find_all(class_='w')
    else:
        body = soup.find_all('body')

    contents=''
    for i in body:
        contents += i.get_text()

    new_contents = re.findall('[가-힣\n. ]', contents)  # 순수 한글,줄바꿈,띄어쓰기만 뽑아냄

    #첫 한글 나오기 전 문자 제거
    for i in range(len(new_contents)):
        if(new_contents[i].isalpha()):
            break
        else:
            new_contents[i]=''

    #한 문장 끝나면 줄바꿈
    for i in range(len(new_contents)):
        if(new_contents[i] == '.'):
            new_contents[i] = ''
            if(i != len(new_contents)-1):
                if (new_contents[i + 1] != '\n'):
                    new_contents[i + 1] = '\n'


    #\n 나온 후 한글 나옴
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

    #마지막 줄바꿈 제거 후 끝에 줄바꿈 넣어줌
    for i in range(len(new_contents)-1,0,-1):
        if(new_contents[i]=='\n'):
            new_contents[i]=''
        if(new_contents[i].isalpha()):
            new_contents.append('\n')
            break

    #텍스트 파일에 저장
    f = open('crawling.txt', 'a')
    f.write("".join(new_contents)) #리스트에 저장된 각 문자들을 문자열로 결합
    f.close()
