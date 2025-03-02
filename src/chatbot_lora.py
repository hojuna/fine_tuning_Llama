import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from .utils import CHAT_TEMPLATE  


def chatbot():
    adapter_save_dir = "output/adapter"

    # PEFT 설정과 base model 로드
    peft_config = PeftConfig.from_pretrained(adapter_save_dir)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, adapter_save_dir)
    tokenizer = AutoTokenizer.from_pretrained("output/tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    system_instruction = (
        "당신은 한국어 작문 스타일에 대한 전문 지식을 가진 어시스턴트입니다. "
        "사용자의 질문에 대해 상세하고 창의적이며 명확한 답변을 제공해 주세요."
    )

    # 파이프라인은 루프 밖에서 한 번 생성 (GPU 사용 가능 시 device=0 지정)
    chat_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1,
    )

    print("\nChatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break

        # 대화 형식을 구성: 시스템, 사용자, 어시스턴트(응답 시작 태그 {% generation %} 포함)
        conversation = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ""}
        ]

        # CHAT_TEMPLATE을 적용해 프롬프트 생성 (토큰화 없이 텍스트 형식)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        response = chat_pipeline(
            prompt,
            max_length=1024,
            do_sample=True,
            temperature=0.7,
        )

        generated_text = response[0]["generated_text"]

        answer = generated_text.strip()

        print(f"Bot: {answer}")

if __name__ == "__main__":
    chatbot()
