import requests
from bs4 import BeautifulSoup
import re

# a function to crawl page from url
def crawling(url):
    if (url != ''):
        resp = requests.get(url)

        if (resp.status_code == 200):
            soup = BeautifulSoup(resp.text, 'html.parser')

            if 'namu.wiki' in url:
                body = soup.find_all(class_='w')
            else:
                body = soup.find_all('body')

            contents = ''
            for i in body:
                contents += i.get_text()

            new_contents = re.findall('[가-힣\n. ]', contents)  # 순수 한글,줄바꿈,띄어쓰기만 뽑아냄

            # 파일 첫 문자는 한글
            for i in range(len(new_contents)):
                if (new_contents[i].isalpha()):
                    break
                else:
                    new_contents[i] = ''

            # 한 문장 끝나면 줄바꿈
            for i in range(len(new_contents)):
                if (new_contents[i] == '.'):
                    new_contents[i] = ''
                    if (i != len(new_contents) - 1):
                        if (new_contents[i + 1] != '\n'):
                            new_contents[i + 1] = '\n'

            # \n후 한글 제외한 문자 삭제
            for i in range(len(new_contents)):
                if (new_contents[i] == '\n'):
                    for k in range(len(new_contents) - i - 1):
                        if (new_contents[i + k + 1].isalpha()):
                            break
                        else:
                            new_contents[i + k + 1] = ""

            # 띄어쓰기 중복 제거
            for i in range(len(new_contents) - 1):
                if (new_contents[i] == ' '):
                    if (new_contents[i + 1] == ' '):
                        new_contents[i] = ""

            # 파일 끝에 줄바꿈 하나 추가
            for i in range(len(new_contents) - 1, 0, -1):
                if (new_contents[i] == '\n'):
                    new_contents[i] = ''
                if (new_contents[i].isalpha()):
                    new_contents.append('\n')
                    break

            # 텍스트 파일에 저장
            f = open('crawling.txt', 'a')
            f.write("".join(new_contents).replace(' ', '\n').replace('\n\n', '\n'))
            f.close()
            generate_dictionary()


# a function to make two dictionary files
def generate_dictionary():
    str2int = {}
    str2int["_"] = 0
    int2str = {}
    int2str[0] = "_"

    f = open('crawling.txt', 'r', encoding='utf-8')
    text = f.read()

    for c in text:
        if c not in str2int:
            if (c != '\n'):
                str2int[c] = len(str2int)
                int2str[len(int2str)] = c

    str2int_f = open('str2int.txt', 'w', encoding='utf-8')
    str2int_f.write(str(str2int).replace('\'', '\"'))

    int2str_f = open('int2str.txt', 'w', encoding='utf-8')
    int2str_f.write(str(int2str).replace('\'', '\"'))

    f.close()
    str2int_f.close()
    int2str_f.close()


generate_dictionary()