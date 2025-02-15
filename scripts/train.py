import argparse
import json
import logging
import math
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

import wandb  # wandb 임포트

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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
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
        default=1024,
        help="Max sequence length for tokenization",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=True,
        help="Whether to trust remote code",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        type=bool,
        default=True,
        help="Enable low CPU memory usage",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        type=bool,
        default=True,
        help="Use the slow version of the tokenizer",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/huggingface_cache/datasets/coastral___korean-writing-style-instruct",
        help="Dataset path",
    )
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--checkpoint_steps", type=int, default=250)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument(
        "--run_name",
        type=str,
        default="20250214_lora",
        help="Name for the wandb run",
    )
    return parser.parse_args()


def generate_prompt(conversation):
    result = ""
    for turn in conversation:
        if turn["from"] == "human":
            result += f"### Instruction:\n{turn['value']}\n\n"
        elif turn["from"] == "assistant":
            result += f"### Response:\n{turn['value']}\n\n"
    return result.strip()


def load_model_and_tokenizer(args, config):
    """모델과 토크나이저를 로드합니다."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        model_max_length=args.max_length,
        use_slow_tokenizer=args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()  
    model.gradient_checkpointing_enable()
    return model, tokenizer


def load_and_preprocess_data(
    args, accelerator, tokenizer, train_ratio=0.8, val_ratio=0.1
):
    """데이터셋 로드 및 전처리 함수"""

    if args.local:
        dataset = load_dataset(os.path.abspath(args.dataset_path), keep_in_memory=False)
    else:
        dataset = load_dataset("coastral/korean-writing-style-instruct")

    # 데이터셋 크기 로깅
    logger.info(f"원본 데이터셋 크기: {len(dataset['train'])}")

    # 전체 데이터셋을 랜덤하게 섞은 후 분할
    shuffled_dataset = dataset["train"].shuffle(seed=42)

    total_size = len(shuffled_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    data_train = shuffled_dataset.select(range(train_size))
    data_val = shuffled_dataset.select(range(train_size, train_size + val_size))
    data_test = shuffled_dataset.select(range(train_size + val_size, total_size))

    logger.info(
        f"분할 후 크기 - Train: {len(data_train)}, Val: {len(data_val)}, Test: {len(data_test)}"
    )

    def _preprocess_function(examples):
        # conversations 필드에서 직접 접근
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
        tokenized_train = data_train.map(
            _preprocess_function,
            batched=True,
            remove_columns=data_train.column_names,
        )
        tokenized_val = data_val.map(
            _preprocess_function, batched=True, remove_columns=data_val.column_names
        )
        tokenized_test = data_test.map(
            _preprocess_function, batched=True, remove_columns=data_test.column_names
        )

    return tokenized_train, tokenized_val, tokenized_test


def create_dataloaders(train_dataset, val_dataset, test_dataset, args, tokenizer):
    """DataLoader 생성 (collate_fn 포함)"""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
    )
    return train_dataloader, eval_dataloader, test_dataloader


def train(
    args, accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
):
    model.eval()
    first_loss, first_perplexity = evaluate(args, accelerator, model, eval_dataloader)
    logger.info(
        "Epoch 0 - Train Loss: N/A, Eval Loss: %.4f, Perplexity: %.4f",
        first_loss.item(),
        first_perplexity,
    )
    if accelerator.is_main_process:
        wandb.log(
            {
                "epoch": 0,
                "train_loss": first_loss.item(),
                "eval_loss": first_loss.item(),
            }
        )
    model.train()

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training Steps",
    )
    completed_steps = 0

    for epoch in range(args.num_train_epochs):

        epoch_train_loss = 0.0
        num_batches = 0

        if hasattr(train_dataloader, "sampler") and hasattr(
            train_dataloader.sampler, "set_epoch"
        ):
            train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                epoch_train_loss += loss.item()
                num_batches += 1

            completed_steps += 1
            progress_bar.update(1)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "global_step": completed_steps,
                    }
                )

                if completed_steps % args.checkpoint_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        logger.info("Checkpoint saved at step %d", completed_steps)

            if completed_steps % args.eval_interval == 0:

                model.eval()
                eval_loss, perplexity = evaluate(
                        args, accelerator, model, eval_dataloader
                    )
                model.train()


                if accelerator.is_main_process:
                    logger.info("Evaluating at step %d", completed_steps)
                    # 현재까지의 평균 train loss 계산
                    avg_train_loss = (
                        epoch_train_loss / num_batches
                        if num_batches > 0
                        else float("inf")
                    )
                    logger.info(
                        "Epoch %d - Train Loss: %.4f, Eval Loss: %.4f, Perplexity: %.4f",
                        epoch,
                        avg_train_loss,
                        eval_loss,
                        perplexity,
                    )

                    # WandB 로깅
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

        ckpt_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
        accelerator.save_state(ckpt_dir)
        if accelerator.is_main_process:
            logger.info("Final checkpoint of epoch saved at step %d", completed_steps)

        # 에포크 종료 시 평균 loss 계산
        avg_train_loss = (
            epoch_train_loss / num_batches if num_batches > 0 else float("inf")
        )

    # 학습 종료 시 마지막 체크포인트 저장
    ckpt_dir = os.path.join(args.output_dir, f"final_checkpoint")
    accelerator.save_state(ckpt_dir)
    if accelerator.is_main_process:
        logger.info("Final checkpoint saved at step %d", completed_steps)


def evaluate(args, accelerator, model, eval_dataloader):

    losses = []
    for batch in tqdm(eval_dataloader, desc="Evaluating",disable=not accelerator.is_local_main_process):
        with torch.no_grad():

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size))
            )
    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    torch.cuda.empty_cache()
    return eval_loss, perplexity


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16"
    )
    accelerator.init_trackers("fine-tune-llama", config=vars(args))
    set_seed(42)

    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 메인 프로세스에서만 WandB 초기화
    if accelerator.is_main_process:
        wandb.init(
            project="fine-tune-llama_02_14",
            name=args.run_name,
            config=vars(args)
        )

    config = AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    model, tokenizer = load_model_and_tokenizer(args, config)

    tokenized_train, tokenized_val, tokenized_test = load_and_preprocess_data(
        args=args, accelerator=accelerator, tokenizer=tokenizer
    )
    train_dataloader, eval_dataloader, test_dataloader = create_dataloaders(
        tokenized_train, tokenized_val, tokenized_test, args, tokenizer
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

    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(

        optimizer,
        round(args.max_train_steps * args.lr_warmup_ratio),
        args.max_train_steps,
    )
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
        )
    )

    train(
        args,
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
    )
    test_loss, test_perplexity = evaluate(args, accelerator, model, test_dataloader)
    logger.info("Test Loss: %.4f, Perplexity: %.4f", test_loss.item(), test_perplexity)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        wandb.finish()  # WandB 세션 종료


if __name__ == "__main__":
    main()
