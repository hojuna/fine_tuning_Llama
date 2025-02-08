import argparse
import json
import logging
import math
import os

import torch
import wandb  # wandb 임포트
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

from accelerate import Accelerator
from accelerate.utils import set_seed

# 캐시 경로 설정
CACHE_DIR = "/home/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
HF_HOME = os.path.join(CACHE_DIR, "hub")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with Accelerate")
    parser.add_argument(
        "--local", action="store_true", help="Use local model and dataset cache"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save checkpoints and final model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model identifier or local path",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for tokenization",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    # checkpointing_steps 관련 옵션은 이제 사용하지 않습니다.
    # parser.add_argument("--checkpointing_steps", type=str, default="epoch")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Whether to trust remote code"
    )
    parser.add_argument(
        "--low_cpu_mem_usage", action="store_true", help="Enable low CPU memory usage"
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Use the slow version of the tokenizer",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args, config):
    """모델과 토크나이저를 로드합니다."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        model_max_length=args.max_length,
        use_slow_tokenizer=args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.local:
        model_path = os.path.join(
            HF_HOME,
            "models--meta-llama--Llama-3.2-1B-Instruct",
            "snapshots",
            "9213176726f574b556790deb65791e0c5aa438b6",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )

    # 학습 시 cache 사용하지 않음 (inference에서의 캐싱과 차원 불일치 문제 방지)
    model.config.use_cache = False
    return model, tokenizer


def load_and_preprocess_data(args, accelerator, tokenizer):
    """데이터셋 로드 및 전처리 함수"""
    if args.local:
        dataset = load_dataset(
            "coastral/korean-writing-style-instruct",
            cache_dir=os.path.join(CACHE_DIR, "datasets", "comoz_cache"),
        )
    else:
        dataset = load_dataset("coastral/korean-writing-style-instruct")

    # train/validation 분리
    if "validation" in dataset.keys():
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    else:
        split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]

    # 학습 데이터셋을 10,000개로 샘플링 (원하는 경우 무작위 샘플링도 고려할 수 있음)
    if len(train_dataset) > 10000:
        train_dataset = train_dataset.select(range(10000))

    logger.info("Dataset columns: %s", dataset["train"].column_names)

    def generate_prompt(conversation):
        result = ""
        for turn in conversation:
            if turn["from"] == "human":
                result += f"### Instruction:\n{turn['value']}\n\n"
            elif turn["from"] == "assistant":
                result += f"### Response:\n{turn['value']}\n\n"
        return result.strip()

    def _preprocess_function(examples):
        prompts = [generate_prompt(conv) for conv in examples["conversations"]]
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        for i in range(len(tokenized["attention_mask"])):
            tokenized["labels"][i] = [
                -100 if mask == 0 else token
                for mask, token in zip(
                    tokenized["attention_mask"][i], tokenized["labels"][i]
                )
            ]
        return tokenized

    with accelerator.main_process_first():
        tokenized_train = train_dataset.map(
            _preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        tokenized_val = val_dataset.map(
            _preprocess_function, batched=True, remove_columns=val_dataset.column_names
        )

    return tokenized_train, tokenized_val


def create_dataloaders(train_dataset, val_dataset, args):
    """DataLoader 생성 (collate_fn 포함)"""
    collate_fn = lambda batch: {
        key: torch.tensor([sample[key] for sample in batch]) for key in batch[0].keys()
    }

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collate_fn,
    )
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collate_fn,
    )
    return train_dataloader, eval_dataloader


def train_and_evaluate(
    args, accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
):
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss.mean() / args.gradient_accumulation_steps
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                epoch_train_loss += loss.item()
                num_batches += 1

            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)

                # 1000 step마다 checkpoint 저장 (메인 프로세스에서만)
                if completed_steps % 1000 == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        logger.info("Checkpoint saved at step %d", completed_steps)

            if completed_steps >= args.max_train_steps:
                break

        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else None

        model.eval()
        losses = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        logger.info(
            "Epoch %d - Train Loss: %.4f, Eval Loss: %.4f, Perplexity: %.4f",
            epoch,
            avg_train_loss,
            eval_loss,
            perplexity,
        )

        # WandB에 로그 기록 (메인 프로세스에서만)
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "eval_loss": eval_loss.item(),
                    "perplexity": perplexity,
                    "completed_steps": completed_steps,
                }
            )

        if completed_steps >= args.max_train_steps:
            break

    return perplexity


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    accelerator.init_trackers("fine-tune-llama", config=vars(args))
    set_seed(42)

    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 메인 프로세스에서만 WandB 초기화
    if accelerator.is_main_process:
        wandb.init(project="fine-tune-llama", config=vars(args))

    config = AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    model, tokenizer = load_model_and_tokenizer(args, config)

    tokenized_train, tokenized_val = load_and_preprocess_data(
        args=args, accelerator=accelerator, tokenizer=tokenizer
    )
    train_dataloader, eval_dataloader = create_dataloaders(
        tokenized_train, tokenized_val, args
    )

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    perplexity = train_and_evaluate(
        args,
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
    )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity}, f)
        wandb.finish()  # WandB 세션 종료


if __name__ == "__main__":
    main()
