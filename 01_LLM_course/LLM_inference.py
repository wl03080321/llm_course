from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import threading
from typing import List, Dict, Union
from pathlib import Path as path

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 啟用 8-bit 量化
)

quantization_config = None
base_dir = path(__file__).parent.parent
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config if quantization_config else None,
        trust_remote_code=True,
        device_map='auto',
        use_auth_token=True, # 需要 Hugging Face 的訪問令牌
        cache_dir= base_dir / "cache"
    )

tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map='auto',
        use_auth_token=True,
        cache_dir= base_dir / "cache"
    )


def chat_stream(input_data:Union[List[Dict],str]) -> str:
    if isinstance(input_data, str):
        inputs = tokenizer(input_data, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
    else:
        input_ids = tokenizer.apply_chat_template(
            input_data,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)                   

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,           # 跳過提示詞
        skip_special_tokens=True    # 跳過特殊 token
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            input_ids= input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            streamer=streamer
        )
    )

    model.eval()
    thread.start()
    response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        response += new_text
    print()
    return response

if __name__ == "__main__":
    history = []
    print("開始對話，輸入 'exit' 離開")
    while True:
        user_input = input("你：")
        if user_input.lower() == "exit":
            break
        history.append({"role": "user", "content": user_input})
        response = chat_stream(history)
        history.append({"role": "assistant", "content": response})