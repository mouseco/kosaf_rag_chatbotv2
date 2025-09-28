# 🤖 KOSAF RAG 챗봇

한국장학재단 규정 기반 RAG(Retrieval-Augmented Generation) 챗봇입니다. 등록된 규정 문서를 기반으로 정확하고 상세한 답변을 제공합니다.

## ✨ 주요 기능

- 📚 **문서 기반 RAG**: 등록된 규정 문서를 검색하여 정확한 답변 제공
- 🛡️ **개인정보 보호**: 주민번호, 전화번호, 이메일 등 개인정보 자동 감지 및 차단
- 🔒 **보안 강화**: 민감 키워드 검사 및 실시간 필터링
- 📄 **문서 관리**: PDF 문서 업로드, 삭제 및 관리 기능
- 💬 **대화형 UI**: 직관적인 채팅 인터페이스
- 📊 **문서 소스 표시**: 답변 근거가 된 문서 출처 명시

## 🛡️ 보안 기능

### 개인정보 보호
- **주민등록번호** 패턴 감지 및 차단
- **전화번호** (휴대폰, 일반전화) 패턴 감지
- **이메일 주소** 패턴 감지
- **신용카드번호** 패턴 감지
- **계좌번호** 패턴 감지
- **사업자등록번호** 패턴 감지

### 민감 정보 검사
- 이름, 주소, 비밀번호 등 민감 키워드 검사
- 사용자 확인 후 처리 진행
- 입력 내용 자동 정화

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/kosaf-rag-chatbot.git
cd kosaf-rag-chatbot
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux  
source .venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
GOOGLE_API_KEY=your_google_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

### 5. Streamlit 앱 실행
```bash
streamlit run app.py
```

## 🔧 Streamlit Cloud 배포

### 1. GitHub 저장소 준비
- 코드를 GitHub에 푸시
- Public 저장소로 설정

### 2. Streamlit Cloud에서 배포
1. [Streamlit Cloud](https://streamlit.io/cloud)에 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 저장소와 브랜치 선택
5. Main file path: `app.py`
6. Advanced settings에서 환경 변수 설정

### 3. 환경 변수 설정 (Streamlit Cloud)
Secrets 섹션에 다음과 같이 추가:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
SUPABASE_URL = "your_supabase_url_here"  
SUPABASE_KEY = "your_supabase_anon_key_here"
```

## 📖 사용 방법

### 1. 문서 업로드
- 사이드바에서 "📄 문서 관리" 섹션 이용
- PDF 파일을 드래그 앤 드롭으로 업로드
- 업로드된 문서는 자동으로 벡터화되어 저장

### 2. 문서 선택
- "📚 문서 선택" 섹션에서 검색에 사용할 문서 선택
- 여러 문서 동시 선택 가능

### 3. 질문하기
- 하단 채팅 입력창에 질문 입력
- AI가 선택된 문서를 기반으로 답변 생성
- 답변 하단에 참고된 문서 출처 표시

## 🛠️ 주요 구성 요소

### 1. 백엔드 (Supabase)
- **documents** 테이블: 문서 메타데이터 저장
- **document_chunks** 테이블: 문서 청크 및 임베딩 저장
- **pgvector** 확장: 벡터 유사도 검색

### 2. AI 모델
- **Google Gemini 2.5 Flash**: 텍스트 생성
- **text-embedding-004**: 문서 임베딩

### 3. 보안 시스템
- 정규표현식 기반 개인정보 패턴 감지
- 민감 키워드 필터링
- 실시간 입력 검증

## 📋 필요한 서비스

### 1. Google AI Studio
- Gemini API 키 발급
- [Google AI Studio](https://aistudio.google.com/) 접속

### 2. Supabase
- 데이터베이스 및 벡터 저장소
- [Supabase](https://supabase.com/) 계정 생성

### 3. Streamlit Cloud
- 웹 앱 배포 플랫폼
- [Streamlit Cloud](https://streamlit.io/cloud) 계정 생성

## 🔍 프로젝트 구조

```
kosaf-rag-chatbot/
├── app.py                 # 메인 Streamlit 앱
├── requirements.txt       # Python 패키지 의존성
├── .gitignore            # Git 무시 파일 목록
├── .env                  # 환경 변수 (로컬)
├── upload_script.py      # 문서 업로드 스크립트
├── delete_documents.py   # 문서 삭제 스크립트
└── README.md            # 프로젝트 문서
```

## 🚨 주의사항

- **개인정보 입력 금지**: 주민번호, 전화번호 등 개인정보 입력 시 자동 차단
- **API 키 보안**: `.env` 파일은 절대 공개 저장소에 업로드하지 마세요
- **문서 관리**: 민감한 문서는 업로드하지 마세요
- **사용량 제한**: Google API 및 Supabase 사용량 제한에 주의하세요

## 🤝 기여하기

버그 리포트나 기능 제안은 GitHub Issues를 통해 해주세요.

## 📄 라이선스

MIT License

---

**개발자**: mouseco  
**버전**: 1.0.0  
**최종 업데이트**: 2025년 9월 28일