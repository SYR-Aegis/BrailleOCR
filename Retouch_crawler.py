f = open('SYR_crawling.txt', 'r')
new_f = open('new_crawling.txt','w')

# 텍스트 단어별 분리
text = f.read()
text = text.replace(' ','\n')
text = text.replace('\n\n','\n')
new_text = text.split('\n')

# 글자 딕셔너리 파일 생성
data_dict = {}
for idx in range(len(new_text)):
    if new_text[idx] not in data_dict:
        data_dict[new_text[idx]] = idx
    new_f.write(new_text[idx] + '\n')

data_f = open('dict_file.txt',w)
data_f.write(data_dict)

f.close()
new_f.close()
data_f.close()




