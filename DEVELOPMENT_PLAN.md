# NBTM (Nonparametric Bayesian Topic Modeling) 개발 계획서

## 프로젝트 개요

### 목적
단일 Jupyter 노트북으로 구현된 Gibbs Sampling LDA를 전문적인 연구용 Python 패키지로 재구성하고, 다양한 토픽 모델링 알고리즘을 비교 실험할 수 있는 프레임워크 구축

### 주요 목표
1. 모듈화된 프로젝트 구조 (src-layout)
2. 다중 알고리즘 지원 (LDA, HDP, CTM 등)
3. CLI 기반 실험 관리
4. 포괄적인 평가 지표 및 시각화
5. 재현 가능한 실험 환경

---

## 기술 스택

| 카테고리 | 기술 |
|---------|------|
| 언어 | Python 3.10+ |
| 패키징 | pyproject.toml + hatchling |
| 설정 관리 | YAML + Dataclass |
| CLI | Click |
| 시각화 | Matplotlib, Seaborn, WordCloud |
| 테스트 | pytest |
| 코드 품질 | black, ruff, mypy |
| 실험 추적 | WandB (선택) |

---

## 프로젝트 구조

```
nonparametric/
├── pyproject.toml              # 패키지 설정
├── README.md                   # 프로젝트 소개
├── DEVELOPMENT_PLAN.md         # 본 문서
├── LICENSE
├── .gitignore
│
├── configs/                    # YAML 설정 파일
│   ├── default.yaml
│   └── models/
│       ├── lda_gibbs.yaml
│       ├── lda_vi.yaml
│       ├── hdp.yaml
│       └── ctm.yaml
│
├── data/                       # 데이터 디렉토리
│   ├── raw/
│   └── processed/
│
├── outputs/                    # 출력 디렉토리
│   ├── models/
│   ├── logs/
│   └── visualizations/
│
├── notebooks/                  # Jupyter 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_lda_tutorial.ipynb
│   ├── 03_hdp_tutorial.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_visualization_examples.ipynb
│
├── docs/                       # 문서
│   ├── getting_started.md
│   ├── algorithms.md
│   ├── configuration.md
│   └── evaluation.md
│
├── scripts/                    # 유틸리티 스크립트
│   ├── download_data.py
│   └── run_experiments.sh
│
├── tests/                      # 테스트
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models/
│   ├── test_data/
│   └── test_evaluation/
│
└── src/nbtm/                   # 메인 패키지
    ├── __init__.py
    ├── config.py               # 설정 시스템
    ├── cli.py                  # CLI 인터페이스
    │
    ├── data/                   # 데이터 처리
    │   ├── __init__.py
    │   ├── corpus.py
    │   ├── preprocessing.py
    │   ├── vocabulary.py
    │   └── dataset.py
    │
    ├── models/                 # 토픽 모델
    │   ├── __init__.py
    │   ├── base.py             # 추상 베이스 클래스
    │   ├── registry.py         # 모델 레지스트리
    │   ├── lda_gibbs.py        # Gibbs Sampling LDA
    │   ├── lda_vi.py           # Variational LDA
    │   ├── hdp.py              # HDP
    │   └── ctm.py              # CTM
    │
    ├── training/               # 학습 인프라
    │   ├── __init__.py
    │   ├── trainer.py
    │   ├── callbacks.py
    │   └── metrics.py
    │
    ├── evaluation/             # 평가
    │   ├── __init__.py
    │   ├── coherence.py
    │   ├── perplexity.py
    │   ├── diversity.py
    │   └── benchmark.py
    │
    ├── visualization/          # 시각화
    │   ├── __init__.py
    │   ├── topic_words.py
    │   ├── document_topics.py
    │   ├── wordcloud.py
    │   ├── convergence.py
    │   └── comparison.py
    │
    └── utils/                  # 유틸리티
        ├── __init__.py
        ├── seed.py
        ├── logging.py
        ├── checkpoint.py
        └── experiment.py
```

---

## 지원 알고리즘

### 1. Gibbs Sampling LDA (현재 구현)
- **방법**: Collapsed Gibbs Sampling
- **특징**: 직관적, 구현 간단, 수렴 보장
- **하이퍼파라미터**: K (토픽 수), α (문서-토픽), β (토픽-단어)

### 2. Variational Inference LDA
- **방법**: Mean-field Variational Bayes
- **특징**: 결정론적, 대규모 데이터에 효율적
- **장점**: 빠른 수렴, 온라인 학습 가능

### 3. Hierarchical Dirichlet Process (HDP)
- **방법**: Chinese Restaurant Franchise / Direct Assignment
- **특징**: 비모수적, 토픽 수 자동 추론
- **하이퍼파라미터**: γ (글로벌 DP), α₀ (로컬 DP)

### 4. Correlated Topic Model (CTM)
- **방법**: Variational EM
- **특징**: 토픽 간 상관관계 모델링
- **분포**: Logistic Normal (Dirichlet 대신)

---

## 구현 단계 (커밋 계획)

### Phase 1: 프로젝트 초기화
**커밋 메시지**: `init: project structure and packaging setup`

- [x] git init
- [ ] 디렉토리 구조 생성
- [ ] pyproject.toml 작성
- [ ] .gitignore 설정
- [ ] README.md 초안

### Phase 2: 설정 시스템
**커밋 메시지**: `feat(config): add dataclass-based configuration system`

- [ ] config.py - Dataclass 정의
  - ModelConfig
  - TrainingConfig
  - DataConfig
  - EvaluationConfig
  - Config (통합)
- [ ] YAML 로드/저장 기능
- [ ] configs/default.yaml

### Phase 3: 데이터 모듈
**커밋 메시지**: `feat(data): add corpus loading and preprocessing`

- [ ] vocabulary.py - 어휘 관리
- [ ] preprocessing.py - 텍스트 전처리
- [ ] corpus.py - 코퍼스 로딩
- [ ] dataset.py - Dataset 추상화

### Phase 4: 모델 베이스 및 레지스트리
**커밋 메시지**: `feat(models): add base class and registry pattern`

- [ ] base.py - BaseTopicModel ABC
  - fit(), transform(), get_topic_words()
  - get_document_topics(), log_likelihood()
- [ ] registry.py - 모델 팩토리
  - @register_model 데코레이터
  - create_model() 함수

### Phase 5: Gibbs LDA 포팅
**커밋 메시지**: `feat(models): port Gibbs Sampling LDA from notebook`

- [ ] lda_gibbs.py
  - 기존 project.ipynb 코드 리팩토링
  - BaseTopicModel 인터페이스 구현
  - 테스트 작성

### Phase 6: 추가 알고리즘
**커밋 메시지**: `feat(models): add LDA-VI, HDP, and CTM implementations`

- [ ] lda_vi.py - Variational LDA
- [ ] hdp.py - HDP
- [ ] ctm.py - CTM
- [ ] 각 알고리즘별 설정 YAML

### Phase 7: 학습 인프라
**커밋 메시지**: `feat(training): add trainer and callback system`

- [ ] trainer.py - 학습 루프 추상화
- [ ] callbacks.py
  - EarlyStopping
  - ModelCheckpoint
  - ProgressLogger
  - WandbLogger
- [ ] metrics.py - 학습 메트릭

### Phase 8: 평가 모듈
**커밋 메시지**: `feat(evaluation): add coherence, perplexity, and diversity metrics`

- [ ] coherence.py
  - UMass, UCI, NPMI, C_V
- [ ] perplexity.py
  - Held-out perplexity
- [ ] diversity.py
  - Topic uniqueness
- [ ] benchmark.py - 벤치마크 러너

### Phase 9: 시각화 모듈
**커밋 메시지**: `feat(visualization): add topic visualization tools`

- [ ] topic_words.py - 토픽-단어 분포
- [ ] document_topics.py - 문서-토픽 히트맵
- [ ] wordcloud.py - 워드 클라우드
- [ ] convergence.py - 수렴 플롯
- [ ] comparison.py - 알고리즘 비교

### Phase 10: CLI 구현
**커밋 메시지**: `feat(cli): add command-line interface`

- [ ] cli.py
  - train 명령
  - evaluate 명령
  - visualize 명령
  - list-models 명령
- [ ] 엔트리 포인트 설정

### Phase 11: 노트북 및 문서화
**커밋 메시지**: `docs: add tutorials and documentation`

- [ ] Jupyter 노트북 예제
- [ ] docs/ 문서
- [ ] README.md 완성

---

## 핵심 인터페이스 설계

### BaseTopicModel (추상 클래스)

```python
class BaseTopicModel(ABC):
    """모든 토픽 모델의 베이스 클래스"""

    @abstractmethod
    def fit(self, documents: List[List[str]],
            num_iterations: int = 1000,
            callbacks: Optional[List] = None) -> "BaseTopicModel":
        """모델 학습"""
        pass

    @abstractmethod
    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """새 문서의 토픽 분포 추론"""
        pass

    @abstractmethod
    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """토픽별 상위 단어 반환"""
        pass

    @abstractmethod
    def get_document_topics(self) -> np.ndarray:
        """문서-토픽 분포 행렬 반환"""
        pass

    @abstractmethod
    def log_likelihood(self) -> float:
        """로그 우도 계산"""
        pass
```

### Config (설정 클래스)

```python
@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvaluationConfig

    seed: int = 42
    output_dir: str = "outputs"
    experiment_name: str = "experiment"

    @classmethod
    def from_yaml(cls, path: str) -> "Config": ...

    def to_yaml(self, path: str) -> None: ...
```

### CLI 사용 예시

```bash
# 모델 학습
nbtm train --config configs/default.yaml --num-topics 10

# 모델 평가
nbtm evaluate --model-path outputs/model.pkl --metrics all

# 시각화 생성
nbtm visualize --model-path outputs/model.pkl --type wordcloud

# 사용 가능한 모델 목록
nbtm list-models
```

---

## 평가 지표

### Topic Coherence
| 지표 | 설명 | 범위 |
|-----|------|-----|
| UMass | 문서 동시 출현 기반 | (-∞, 0] |
| UCI | PMI 기반 | (-∞, +∞) |
| NPMI | 정규화 PMI | [-1, 1] |
| C_V | 통합 지표 | [0, 1] |

### 기타 지표
- **Perplexity**: 낮을수록 좋음
- **Topic Diversity**: 토픽 간 고유성

---

## 의존성

### 필수
```
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
click>=8.1.0
pyyaml>=6.0
tqdm>=4.65.0
wordcloud>=1.9.0
```

### 개발
```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
ruff>=0.1.0
mypy>=1.0.0
```

### 선택
```
jupyter>=1.0.0
wandb>=0.15.0
```

---

## 일정 (예상)

| Phase | 내용 | 예상 커밋 수 |
|-------|------|------------|
| 1 | 프로젝트 초기화 | 1 |
| 2 | 설정 시스템 | 1 |
| 3 | 데이터 모듈 | 1 |
| 4 | 모델 베이스/레지스트리 | 1 |
| 5 | Gibbs LDA 포팅 | 1 |
| 6 | 추가 알고리즘 | 3 |
| 7 | 학습 인프라 | 1 |
| 8 | 평가 모듈 | 1 |
| 9 | 시각화 모듈 | 1 |
| 10 | CLI | 1 |
| 11 | 문서화 | 1 |
| **총계** | | **~13 커밋** |

---

## 참고 자료

### 논문
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. JMLR.
- Teh, Y. W., et al. (2006). Hierarchical Dirichlet Processes. JASA.
- Blei, D. M., & Lafferty, J. D. (2007). Correlated Topic Models. NIPS.

### 참조 프로젝트
- hmcan: 모델 레지스트리, 콜백 시스템 패턴
- NumericalNet: src-layout, 설정 관리 패턴
- hansem-chatbot: 문서화 패턴

---

## 변경 이력

| 날짜 | 버전 | 내용 |
|-----|------|-----|
| 2026-01-07 | 0.1 | 초안 작성 |
