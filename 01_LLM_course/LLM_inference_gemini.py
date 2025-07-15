import os
from typing import List, Dict, Union
from gemini_llm import GeminiLLM
from src.script import load_config

config = load_config()
API_KEY = config.get("gemini", {}).get("api_key", "add_your_api_key_here")
MODEL_NAME = config.get("gemini", {}).get("model_name", "gemini-2.5-flash")

if __name__ == "__main__":
    try:
        if API_KEY == "your_api_key_here":
            print("請設定正確的 GEMINI_API_KEY 環境變數或修改 API_KEY 變數")
            exit(1)
        llm = GeminiLLM(api_key=API_KEY, model=MODEL_NAME)
        print("模型初始化成功！")
    except Exception as e:
        print(f"模型初始化失敗: {e}")
        exit(1)
    conversation_history = []
    temperature = 0.7
    max_tokens = 10240
    conversation_history.append(
        {
            "role": "system",
            "content": "你是一位台灣7-11統一集團的小助理，你必須回答有關7-11超商等相關的問題。",
        }
    )

    while True:
        try:
            response = ""
            user_input = input("\n你：")

            if user_input.lower() == "exit":
                break
            elif user_input.strip() == "":
                continue

            # 添加用戶訊息到歷史
            conversation_history.append({"role": "user", "content": user_input})

            print("模型：", end="", flush=True)

            for chunk in llm.generate(
                messages=conversation_history,
                temperature=temperature,
                max_new_tokens=max_tokens,
            ):
                response += chunk
                print(chunk, end="", flush=True)

            # 添加助手回應到歷史
            conversation_history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\n發生錯誤: {e}")
