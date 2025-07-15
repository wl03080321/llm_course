from transformers import AutoTokenizer
from pathlib import Path as path

base_dir = path(__file__).parent.parent
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True,
    use_auth_token=True,
    cache_dir=base_dir / "cache",
)
# taide/Llama-3.1-TAIDE-LX-8B-Chat
# deepseek-ai/DeepSeek-R1-0528
# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Llama-2-7b-chat-hf
# Qwen/Qwen2.5-7B-Instruct-1M
# Qwen/Qwen2.5-1.5B-Instruct

# # 2. 原本文字
text = "Hello ,How are you? 你好，你最近過得怎麼樣？"
print("=" * 50)
print("原本文字：", text)
print(len(text), "字元")
print("=" * 50)

# 3. 文字轉換成 token
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)
print("Token 列表：", tokens)
print("Token IDs：", token_ids)
print(len(token_ids), "個 Token IDs")
print("=" * 50)

# 4. 將 token_ids 轉回文字
decoded_text = tokenizer.decode(token_ids)
print("解碼後的文字：", decoded_text)
print("=" * 50)

# 5. 示範 apply_chat_template

# print(tokenizer.chat_template)
if tokenizer.chat_template is not None:
    try:
        chat = [
            {"role": "system", "content": "你是一個友好的聊天助手。"},
            {"role": "user", "content": "請介紹一下你自己。"},
            {"role": "assistant", "content": "你好，我是AI助理，很高興為你服務！"},
        ]
        chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        token = tokenizer.encode(chat_prompt)
        print("Chat Template 結果：")
        print(chat_prompt)
        print("Token：")
        print(token)
        print("=" * 50)
    except Exception as e:
        print("應用 chat template 時發生錯誤：", e)
        print("=" * 50)
else:
    print("此模型並未定義chat_template。")
