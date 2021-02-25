# TLGAN

tlgan.py 스크립트를 실행하면 학습을 사용할 수 있다. 

1. generate_datasets.py 로 images 폴더 안에 이미지를 생성

2. generate_gau_map.py로 gaussian map 생성
3. tlgan.py스크립트로 학습



**사용되는 인자들**  

| 인자명       | 용도                            | 기본값         |
| ------------ | ------------------------------- | -------------- |
| -model_path  | 파라미터 저장 위치              | ./tlgan_model/ |
| --data_path  | datatsets의 위치                | ../../data     |
| --epoch      | 학습 에폭                       | 1000           |
| --load_model | 기 학습 이미지 로드(model_path) | False          |
| --batch_size | 이미지 배치 사이즈              | 8              |
|              |                                 |                |

