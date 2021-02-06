# Dataset 생성 방법

generate_dataset.py 스크립트를 실행하면  

images 폴더 안에 TLGAN, CRNN에 사용될 이미지가 생성된다  

**사용되는 인자들**  

| 인자명            | 용도                                              | 기본값          |
| ----------------- | ------------------------------------------------- | --------------- |
| --text_file_path  | 샘플 이미지 생성을 위한 한글 텍스트 파일로의 경로 | ./              |
| --text_file_name  | 샘플 이미지 생성을 위한 한글 텍스트 파일의 이름   | sample_text.txt |
| --TLGAN_save_path | TLGAN용 이미지를 저장할 경로                      | ./images/TLGAN  |
| --CRNN_save_path  | CRNN용 이미지를 저장할 경로                       | ./images/CRNN   |
| --n_text          | 하나의 이미지에 넣을 단어의 수                    | 4               |
| --simple          | 단어의 개수가 너무 많아 일부만 사용하고 싶은 경우 | True            |

### TLGAN에 사용되는 Gaussian cylindrical map 생성방법

generate_gau_map.py를 실행  

*담당자 기술*