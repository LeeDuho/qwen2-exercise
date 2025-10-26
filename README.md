### Qwen2 Implementation Project
### KNU_DLI_LAB / DUHO LEE 

## 프로젝트 개요
Qwen2 모델을 처음부터 구현하는 프로젝트입니다

## 프로젝트 일정
2025-10-19 ~ 2025-10-30

## 현재 예상 진행 계획
- [ ] 프로젝트 구조 설정
- [ ] 개발 환경 구성
- [ ] vllm api 클라이언트 구현
- [ ] MMLU 평가 구현

## 현재 구현 부분

- 2025-10-19  
ssh 키 생성 및 KNU-DLILab-server 로컬 연결, 비밀번호 변경 완료  
프로젝트 디렉토리 생성 및 git clone 완료  
가상환경 생성 완료

- 2025-10-25  
테스트 스크립트 작성  
8000 포트 에러  
63700 포트에 Qwen/Qwen2.5-3B 모델 서빙 중인 것 확인  
api client, data, utils 로 폴더 구조 생성  
기존 가상환경에서 uv로 버전 관리 하도록 변경  
scripts 작성하여 request 시 completion 오는 것 확인  
mmlu 데이터셋에 대해 학습  
허깅페이스 사용하여 mmlu 데이터셋 load 하고 데이터 구조 확인  

- 2025-10-26  
mmlu 데이터셋 로딩하고 mmlu 포맷에 맞게 question prompt 작성을 위한 MMLUUtils 유틸 생성 v1  
MMLUUtils, Qwen2Client 통하여 mmlu 데이터셋, qwen2 모델 활용한 answer과 정답 answer 비교 (mmlu_utils_test 테스트 코드)  


## 실행 방법
```bash
# 가상환경 설정
python -m venv qwen2_env
source qwen2_env/bin/activate
pip install -r requirements.txt
```

## 참고 자료
- [HuggingFace Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)
