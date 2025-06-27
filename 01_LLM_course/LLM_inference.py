from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline, BitsAndBytesConfig
import threading
import torch
from typing import List, Dict, Optional, Union
from pathlib import Path as path

quantization_config = None
base_dir = path(__file__).parent.parent
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # 啟用 8-bit 量化
#     # bnb_4bit_quant_type="nf4",  # 使用 NF4 量化類型
#     # bnb_4bit_use_double_quant=True,  # 啟用雙重量化
#     # bnb_4bit_compute_dtype=torch.float16,  # 計算時使用 float16
#     # bnb_4bit_use_quantized_attention=True,  # 啟用量化注意力
# )

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                            quantization_config=quantization_config if quantization_config else None,
                                            trust_remote_code=True,
                                            device_map='auto',
                                            use_auth_token=True,
                                            cache_dir= base_dir / "cache")

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                        trust_remote_code=True,
                                        device_map='auto',
                                        use_auth_token=True,
                                        cache_dir= base_dir / "cache")

def run_pipeline_inference(task:str, input_data:Union[List[Dict],str], temperature=0.7):
    try:
        pipe = pipeline(task, 
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=256,
                        temperature=temperature,
                        top_k=50,
                        top_p=0.95,
                        device_map='auto' if model.device.type == 'cuda' else None
                )

        results = pipe(input_data)
        print(results)
        if isinstance(input_data, str):
            response = results[0]['generated_text']
        else:
            response = results[0]['generated_text'][-1]["content"]
    except Exception as e:
        print(f"Pipeline inference error: {e}")
        response = "pipeline發生錯誤。"
    return response


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
    #attention_mask = chat_template.ne(tokenizer.pad_token_id).long()  # pad 為 0，其餘為
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
            #attention_mask=attention_mask
            #pad_token_id=tokenizer.eos_token_id
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
        #response = run_pipeline_inference("text-generation", user_input, temperature=0.7)
        history.append({"role": "assistant", "content": response})