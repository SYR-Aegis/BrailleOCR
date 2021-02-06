f = open('crawling.txt', 'r',encoding='utf-8')
new_f = open('new_crawling.txt','w',encoding='utf-8')

# 텍스트 단어별 분리
text = f.read()
text = text.replace(' ','\n')
text = text.replace('\n\n','\n')
new_text = text.split('\n') #리스트
for i in range(len(new_text)):
    new_f.write(new_text[i] + '\n')

# label 딕셔너리 파일 생성
data_dict = {}
for char in text:
    if char not in data_dict:
        if(char!= '\n'):
            data_dict[char] = len(data_dict)

data_f = open('dict_file.txt','w',encoding='utf-8')
data_f.write(str(data_dict).replace('\'','\"'))


f.close()
new_f.close()
data_f.close()




