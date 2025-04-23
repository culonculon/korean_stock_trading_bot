# 한국 주식 트레이딩 봇

실시간 주가 모니터링, 기술적 지표 계산, 자동 매매 신호 생성 기능을 제공하는 한국 주식 트레이딩 봇입니다.

## 주요 기능

- 실시간 주가 모니터링
- 기술적 지표 분석 (이동평균선, RSI, MACD, 볼린저 밴드, 스토캐스틱)
- 매매 신호 생성
- 백테스팅
- 모의 트레이딩

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/your-username/korean_stock_trading_bot.git
cd korean_stock_trading_bot
```

2. 가상 환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
```bash
cp .env.template .env
# .env 파일을 편집하여 API 키 등의 정보 입력
```

## 실행 방법

```bash
streamlit run app.py
```

## 배포 방법 (Streamlit Cloud)

### 1. GitHub 저장소 설정

1. GitHub에 새 저장소 생성
2. 로컬 저장소를 GitHub에 푸시
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/korean_stock_trading_bot.git
git push -u origin main
```

### 2. Streamlit Cloud 설정

1. [Streamlit Cloud](https://streamlit.io/cloud)에 접속하여 로그인 (GitHub 계정으로 로그인 가능)
2. "New app" 버튼 클릭
3. GitHub 저장소, 브랜치, 메인 파일 경로(app.py) 입력
4. Advanced 설정에서 다음 환경 변수 추가:
   - `KIS_APP_KEY`: 실전 투자 앱키
   - `KIS_APP_SECRET`: 실전 투자 앱 시크릿
   - `KIS_ACC_NO`: 실전 투자 계좌번호
   - `KIS_MOCK_APP_KEY`: 모의 투자 앱키
   - `KIS_MOCK_APP_SECRET`: 모의 투자 앱 시크릿
   - `KIS_MOCK_ACC_NO`: 모의 투자 계좌번호
   - `KIS_USE_MOCK`: True (모의투자 사용 여부)
   - `USE_SESSION_STATE`: True (클라우드 환경에서는 세션 상태 사용)
5. "Deploy" 버튼 클릭

### 3. 배포 후 확인

배포가 완료되면 제공된 URL로 접속하여 앱이 정상적으로 작동하는지 확인합니다.

## 주의사항

- 실제 투자에 사용할 경우 반드시 모의투자로 충분히 테스트한 후 사용하세요.
- API 키와 시크릿 키는 절대 공개 저장소에 업로드하지 마세요.
- Streamlit Cloud에서는 `USE_SESSION_STATE=True`로 설정하여 토큰을 세션 상태에 저장하도록 해야 합니다.

## 외부 개발 환경 설정 (GitHub Codespaces)

GitHub Codespaces를 사용하면 웹 브라우저에서 VS Code와 동일한 개발 환경을 사용하여 어디서든 코드를 수정하고 테스트할 수 있습니다. Streamlit Cloud와 GitHub 저장소가 연동되어 있으므로, Codespaces에서 변경한 내용을 GitHub에 푸시하면 Streamlit Cloud가 자동으로 업데이트됩니다.

### 1. GitHub Codespaces 설정 방법

1. **GitHub 저장소 접속**:
   - GitHub에 로그인하고 한국 주식 트레이딩 봇 저장소로 이동합니다.

2. **Codespaces 생성**:
   - 저장소 페이지에서 "Code" 버튼을 클릭합니다.
   - "Codespaces" 탭을 선택합니다.
   - "Create codespace on main" 버튼을 클릭합니다.

3. **개발 환경 설정**:
   - Codespace가 생성되면 VS Code와 동일한 웹 기반 IDE가 열립니다.
   - 이 프로젝트에는 `.devcontainer/devcontainer.json` 파일이 포함되어 있어 자동으로 개발 환경이 설정됩니다.
   - 필요한 패키지가 자동으로 설치되고 Streamlit 포트(8501)가 자동으로 포워딩됩니다.

4. **환경 변수 설정**:
   - Codespaces에서 환경 변수를 설정하려면 다음 단계를 따릅니다:
     - 왼쪽 하단의 설정 아이콘(⚙️)을 클릭합니다.
     - "Settings"를 선택합니다.
     - "Codespaces"를 검색합니다.
     - "Codespaces: Environment Variables"를 찾아 "Edit in settings.json"을 클릭합니다.
     - 다음과 같이 환경 변수를 추가합니다:
       ```json
       "codespaces.environmentVariables": {
         "KIS_APP_KEY": "your_app_key_here",
         "KIS_APP_SECRET": "your_app_secret_here",
         "KIS_ACC_NO": "your_account_number_here",
         "KIS_MOCK_APP_KEY": "your_mock_app_key_here",
         "KIS_MOCK_APP_SECRET": "your_mock_app_secret_here",
         "KIS_MOCK_ACC_NO": "your_mock_account_number_here",
         "KIS_USE_MOCK": "True",
         "USE_SESSION_STATE": "False"
       }
       ```

### 2. 개발 및 테스트 방법

1. **코드 수정**:
   - Codespaces에서 코드를 자유롭게 수정합니다.

2. **로컬 테스트**:
   - 터미널에서 다음 명령어를 실행하여 Streamlit 앱을 실행합니다:
     ```bash
     streamlit run app.py
     ```
   - Codespaces는 자동으로 포트를 포워딩하여 웹 브라우저에서 앱을 확인할 수 있게 해줍니다.

3. **변경 사항 커밋 및 푸시**:
   - 변경 사항을 테스트한 후, 다음 명령어로 GitHub에 푸시합니다:
     ```bash
     git add .
     git commit -m "Update: 변경 내용 설명"
     git push
     ```

4. **Streamlit Cloud 자동 업데이트**:
   - GitHub에 변경 사항을 푸시하면 Streamlit Cloud가 자동으로 새 버전을 배포합니다.
   - 기본적으로 Streamlit Cloud는 GitHub 저장소의 변경 사항을 감지하여 자동으로 재배포합니다.

## 라이선스

MIT License