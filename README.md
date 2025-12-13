# 20252R0136COSE36203

25-2 COSE362-03 ML Term Project 25조

모든 코드는 colab에서 실행하는 것을 전제로 동작합니다.

### pipeline.ipynb

유튜브 리포트 생성 통합 pipeline.
category, sentimental 모델 로드, 학습해 사용 가능.

모델 파일 링크: [category](https://drive.google.com/file/d/139UACMdcqj5V-iGBB8LZjjd4tCFaTvmx/view?usp=sharing), [sentimental](https://drive.google.com/file/d/1CSZHLOvhcNfRlcJfelGmcb6PuAeSkJTR/view?usp=drive_link )

실행시 의도된 디렉토리 구조

### web_demo_for_colab.ipynb

LLM 리포트 생성웹 데모

defalut [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 사용 시 hugging face 승인 및 토큰 필요(다른 모델 사용 가능, 파일 내 참조).

YouTube API key 필요(get from [here](https://console.developers.google.com)).

실행시 의도된 디렉토리 구조

### youtube_auto_collection.ipynb

유튜브 데이터 수집 코드

YouTube API key 필요(get from [here](https://console.developers.google.com)).

실행시 의도된 디렉토리 구조
