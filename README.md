# LLaMA 파인튜닝 프로젝트

이 프로젝트는 LLaMA 모델을 다양한 방법으로 파인튜닝하기 위한 도구 모음입니다. FSDP(Fully Sharded Data Parallel), LoRA(Low-Rank Adaptation), QLoRA(Quantized LoRA) 등의 기법을 활용하여 효율적인 모델 학습을 지원합니다.

## 📋 목차

- [특징](#특징)
- [요구 사항](#요구-사항)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [프로젝트 구조](#프로젝트-구조)
- [학습 스크립트 설명](#학습-스크립트-설명)
- [챗봇 사용](#챗봇-사용)
- [설정](#설정)

## ✨ 특징

- **다양한 파인튜닝 기법 지원**
  - Full Fine-tuning (FSDP 활용)
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)

- **효율적인 학습**
  - FSDP를 통한 분산 학습
  - Gradient checkpointing으로 메모리 최적화
  - Mixed precision 학습 (FP16/BF16)
  - Gradient accumulation 지원

- **모니터링 및 로깅**
  - Weights & Biases (wandb) 통합
  - TensorBoard 지원
  - 학습 과정 실시간 추적

- **유연한 설정**
  - YAML 기반 설정 파일
  - 커맨드 라인 인자 지원
  - 로컬 및 원격 모델/데이터셋 지원

## 📦 요구 사항

이 프로젝트는 다음 주요 라이브러리를 필요로 합니다:

```
torch
transformers
datasets
accelerate
peft
bitsandbytes
flash-attn
wandb
tensorboard
tqdm
nvitop
```

전체 요구 사항은 `requirement.txt` 파일을 참조하세요.

## 🚀 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/hojuna/fine_tuning_Llama.git
cd fine_tuning_Llama
```

2. 필요한 패키지 설치:
```bash
pip install -r requirement.txt
```

3. (선택사항) Flash Attention 설치:
```bash
pip install flash-attn --no-build-isolation
```

## 💻 사용 방법

### 1. Full Fine-tuning

FSDP를 사용한 전체 파인튜닝:

```bash
accelerate launch --config_file config.yaml scripts/train.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --output_dir output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --learning_rate 3e-5
```

### 2. LoRA Fine-tuning

효율적인 LoRA 파인튜닝:

```bash
accelerate launch --config_file config.yaml scripts/train_lora.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --output_dir output_lora \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32
```

### 3. QLoRA Fine-tuning

양자화된 LoRA 파인튜닝 (메모리 효율적):

```bash
accelerate launch --config_file config.yaml scripts/train_qlora.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --output_dir output_qlora \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32
```

## 📁 프로젝트 구조

```
fine_tuning_Llama/
├── config.yaml              # Accelerate 설정 파일 (FSDP 설정 포함)
├── requirement.txt          # Python 패키지 의존성
├── scripts/                 # 학습 스크립트
│   ├── train.py            # Full fine-tuning 스크립트
│   ├── train_lora.py       # LoRA fine-tuning 스크립트
│   ├── train_qlora.py      # QLoRA fine-tuning 스크립트
│   └── utils.py            # 유틸리티 함수 및 채팅 템플릿
└── src/                     # 소스 코드
    ├── chatbot.py          # Full fine-tuned 모델용 챗봇
    └── chatbot_lora.py     # LoRA fine-tuned 모델용 챗봇
```

## 📚 학습 스크립트 설명

### train.py
- **목적**: 전체 모델 파라미터 파인튜닝
- **특징**: FSDP를 활용한 분산 학습, 높은 성능
- **권장 사용**: GPU 메모리가 충분하고 최상의 성능이 필요한 경우

### train_lora.py
- **목적**: LoRA를 사용한 효율적인 파인튜닝
- **특징**: 적은 파라미터만 학습, 메모리 효율적
- **권장 사용**: 제한된 리소스로 좋은 성능을 얻고자 할 때

### train_qlora.py
- **목적**: 양자화된 LoRA 파인튜닝
- **특징**: 4비트 양자화로 메모리 사용량 최소화
- **권장 사용**: GPU 메모리가 매우 제한적인 환경

## 💬 챗봇 사용

학습된 모델을 대화형 챗봇으로 테스트할 수 있습니다:

### Full Fine-tuned 모델 사용:
```bash
python src/chatbot.py
```

### LoRA Fine-tuned 모델 사용:
```bash
python src/chatbot_lora.py
```

챗봇이 실행되면 사용자 입력을 받아 모델의 응답을 생성합니다. 종료하려면 `exit`를 입력하세요.

## ⚙️ 설정

### config.yaml
Accelerate 설정 파일로 FSDP 파라미터를 정의합니다:
- `distributed_type`: FSDP 사용
- `num_processes`: 사용할 GPU 수
- `mixed_precision`: FP16/BF16 학습
- `fsdp_config`: FSDP 세부 설정 (sharding 전략, offloading 등)

### 주요 학습 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--model_name` | 사용할 모델 이름 | `meta-llama/Llama-3.2-1B-Instruct` |
| `--dataset_name` | 학습 데이터셋 | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` |
| `--output_dir` | 모델 저장 경로 | `output` |
| `--num_train_epochs` | 학습 에폭 수 | `3` |
| `--per_device_train_batch_size` | 디바이스당 배치 크기 | `4` |
| `--gradient_accumulation_steps` | Gradient accumulation 스텝 | `3` |
| `--learning_rate` | 학습률 | `3e-5` |
| `--max_length` | 최대 시퀀스 길이 | `512` |
| `--lora_r` | LoRA rank | `16` |
| `--lora_alpha` | LoRA alpha | `32` |
| `--lora_dropout` | LoRA dropout | `0.1` |

## 📊 모니터링

학습 과정은 Weights & Biases를 통해 모니터링할 수 있습니다:

1. wandb 계정 생성 (https://wandb.ai)
2. 로그인:
```bash
wandb login
```
3. 학습 스크립트 실행 시 자동으로 로그가 wandb에 기록됩니다.

## 🔍 팁과 권장 사항

1. **메모리 부족 시**:
   - `gradient_accumulation_steps` 증가
   - `per_device_train_batch_size` 감소
   - QLoRA 사용 고려
   - `max_length` 감소

2. **학습 속도 향상**:
   - Flash Attention 사용
   - Mixed precision 학습 (FP16/BF16)
   - Gradient checkpointing 활성화

3. **성능 최적화**:
   - 데이터셋에 맞는 `max_length` 설정
   - 적절한 learning rate 튜닝
   - LoRA rank와 alpha 값 조정

## 📝 라이선스

이 프로젝트는 오픈소스이며 자유롭게 사용 및 수정 가능합니다.

## 🤝 기여

버그 리포트, 기능 제안, Pull Request 모두 환영합니다!

## 📧 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 열어주세요.
