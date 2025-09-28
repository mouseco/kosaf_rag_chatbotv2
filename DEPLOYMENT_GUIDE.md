# Streamlit Cloud 배포를 위한 환경 변수 설정 가이드

## 1. 로컬 개발용 (.streamlit/secrets.toml)
로컬에서 테스트할 때는 프로젝트 루트에 `.streamlit` 폴더를 만들고 `secrets.toml` 파일을 생성하세요:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
SUPABASE_URL = "your_supabase_url_here"
SUPABASE_KEY = "your_supabase_anon_key_here"
```

## 2. Streamlit Cloud 배포용 설정
Streamlit Cloud에서 앱을 배포할 때 다음 단계를 따르세요:

### 단계 1: 앱 생성
1. https://streamlit.io/cloud 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. Repository 선택
5. Branch: main (또는 배포할 브랜치)
6. Main file path: `app.py`

### 단계 2: Advanced settings
"Advanced settings" 클릭 후 다음과 같이 입력:

**Secrets** 섹션에 다음 내용을 추가:
```toml
GOOGLE_API_KEY = "your_actual_google_api_key_here"
SUPABASE_URL = "https://your-project-ref.supabase.co"
SUPABASE_KEY = "your_actual_supabase_anon_key_here"
```

### 단계 3: 필수 API 키 발급

#### Google AI Studio API 키
1. https://aistudio.google.com/ 접속
2. "Get API key" 클릭
3. "Create API key" 선택
4. 생성된 API 키를 복사

#### Supabase 설정
1. https://supabase.com/ 접속
2. 새 프로젝트 생성
3. Settings > API 에서 다음 정보 확인:
   - URL: Project URL
   - anon key: anon public key

## 3. 주의사항
- API 키는 절대 GitHub에 업로드하지 마세요
- .streamlit/secrets.toml 파일은 .gitignore에 포함되어 있습니다
- 배포 후 API 사용량을 정기적으로 확인하세요

## 4. 테스트 방법
배포 후 다음 기능들이 정상 작동하는지 확인:
- [ ] 앱 로딩
- [ ] 문서 업로드
- [ ] 문서 검색 및 선택
- [ ] 질문 및 답변 생성
- [ ] 보안 기능 (개인정보 차단)
- [ ] 소스 문서 표시