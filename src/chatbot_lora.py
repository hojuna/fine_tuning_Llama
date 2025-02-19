from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def chatbot():
    adapter_save_dir="adapter"

    peft_config = PeftConfig.from_pretrained(adapter_save_dir)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, adapter_save_dir)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    # pad_token_id 설정
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 파이프라인 생성
    chat_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True,  # Truncation 활성화
    )

    print("\nChatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break

        # 모델 예측을 위한 프롬프트 생성
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        response = chat_pipeline(
            prompt,
            max_length=1024,
            do_sample=True,
            temperature=0.7,
        )

        # 응답에서 프롬프트 부분 제거
        generated_text = response[0]["generated_text"]
        if "### Response:\n" in generated_text:
            answer = generated_text.split("### Response:\n")[1].strip()
        else:
            answer = generated_text.strip()

        print(f"Bot: {answer}")


if __name__ == "__main__":
    chatbot()
