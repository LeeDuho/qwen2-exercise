### Qwen2 Implementation Project
### KNU_DLI_LAB / DUHO LEE 

## 프로젝트 개요
Qwen2 모델을 처음부터 구현하는 프로젝트입니다

## 프로젝트 일정
2025-10-19 ~ 2025-10-30

## 현재 예상 진행 계획
- [ ] 프로젝트 구조 설정
- [ ] 개발 환경 구성
- [ ] 기본 모델 구조 구현
- [ ] Attention 메커니즘 구현
- [ ] 토크나이저 구현
- [ ] 학습 파이프라인 구현
- [ ] 모델 테스트 및 검증

## 현재 구현 부분

- 2025-10-19  
ssh 키 생성 및 KNU-DLILab-server 로컬 연결, 비밀번호 변경 완료  
프로젝트 디렉토리 생성 및 git clone 완료  
가상환경 생성 완료  

## 실행 방법
```bash
# 가상환경 설정
python -m venv qwen2_env
source qwen2_env/bin/activate
pip install -r requirements.txt
```

## 참고 자료
- [HuggingFace Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)
