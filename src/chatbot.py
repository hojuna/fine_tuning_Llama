from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def chatbot():
    # 원본 모델 경로와 파인튜닝된 모델 경로 설정
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 원본 모델
    model_path = "output/step_230"  # 파인튜닝된 모델 경로
    print(f"Loading fine-tuned model from: {model_path}...")

    # Accelerator 초기화
    accelerator = Accelerator()

    # 원본 모델의 토크나이저와 설정을 사용
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,  # 먼저 원본 모델 구조 로드
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 파인튜닝된 가중치 로드
    accelerator.load_state(model_path)
    model = accelerator.unwrap_model(model)

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
            max_length=250,
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
