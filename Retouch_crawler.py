f = open('SYR_crawling.txt', 'r')
new_f = open('new_crawling.txt','w')

text = f.read()
text = text.replace(' ','\n')
text = text.replace('\n\n','\n')

new_text = text.split('\n')

data_dict = {}
for idx in range(len(new_text)):
    if new_text[idx] not in data_dict:
        data_dict[new_text[idx]] = idx
    new_f.write(new_text[idx] + '\n')

f.close()
new_f.close()




