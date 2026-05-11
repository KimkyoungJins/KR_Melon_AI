# Melon Disease Edge Classifier

Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) 기반 참외 잎 질병 5클래스 엣지 분류기.

- 데이터: AIHub 지능형 스마트팜(참외) — `노균병`, `노균병유사`, `흰가루병`, `흰가루병유사`, `정상`
- 워크플로: PyTorch → ONNX (INT8 PTQ) → RUHMI → e² studio/FSP → EK-RA8P1
- 출력: 정지 이미지 5장 분류 + UART 로그 + NPU vs CPU latency 비교

> 셋업 진행 중 — Phase 1 (데이터 준비) 시작 전 단계.

## 빠른 시작 (호스트 PC)

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 디렉토리

```
data/        # raw (AIHub 원본, gitignore) + processed (전처리 산출)
models/      # 학습/양자화 모델 산출
scripts/     # 데이터·학습·평가·양자화 스크립트
firmware/    # RUHMI 산출 + e² studio 소스
docs/        # 다이어그램·그래프
```

## 진행 상황

- [x] 프로젝트 골격
- [ ] Phase 1 — 데이터 준비
- [ ] Phase 2 — 모델 학습
- [ ] Phase 3 — 양자화 + RUHMI 컴파일
- [ ] Phase 4 — 펌웨어 통합
- [ ] Phase 5 — 검증 + 문서화
