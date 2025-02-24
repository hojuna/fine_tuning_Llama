from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig


def chatbot():
    adapter_save_dir = "output/adapter"

    # PEFT 설정과 base model 로드
    peft_config = PeftConfig.from_pretrained(adapter_save_dir)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, adapter_save_dir)
    tokenizer = AutoTokenizer.from_pretrained("output/tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 시스템 지침(시스템 역할) 설정
    system_instruction = (
        "당신은 한국어 작문 스타일에 대한 전문 지식을 가진 어시스턴트입니다. "
        "사용자의 질문에 대해 상세하고 창의적이며 명확한 답변을 제공해 주세요."
    )

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

        # 프롬프트 생성: 시스템, 사용자, 어시스턴트 부분을 명확히 구분하고 어시스턴트 응답 시작에 {% generation %} 태그 추가
        prompt = (
            f"### System:\n{system_instruction}\n\n"
            f"### User:\n{user_input}\n\n"
            f"### Assistant:\n{{% generation %}}"
        )
        response = chat_pipeline(
            prompt,
            max_length=1024,
            do_sample=True,
            temperature=0.7,
        )

        generated_text = response[0]["generated_text"]

        # 어시스턴트 응답 부분만 추출 (태그 이후의 텍스트)
        if "{% generation %}" in generated_text:
            answer = generated_text.split("{% generation %}", 1)[1].strip()
        else:
            answer = generated_text.strip()

        print(f"Bot: {answer}")


if __name__ == "__main__":
    chatbot()
