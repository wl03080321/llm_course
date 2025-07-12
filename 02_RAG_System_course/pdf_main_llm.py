from src.vectordatabase import load_chroma_vectorstore
from src.embedding import load_embedding_model
from src.pdf_process import load_pdf, split_documents
from src.LLM_inference import LLMInference
from src.gemini_llm import GeminiLLM
from src.script import load_config
from pathlib import Path as path
from transformers import BitsAndBytesConfig


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 啟用 8-bit 量化
)


base_dir = path(__file__).parent.parent
vector_store_path = base_dir / "vectorstore"
model_path = base_dir / "cache"
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embedding_model = load_embedding_model(
    model_name=embedding_model_name,
    cache_folder=str(model_path)
)

config = load_config()
choice_llm = config.get("choice_llm", "google")

if choice_llm == "google":
    API_KEY = config.get("gemini", {}).get("api_key", "add_your_api_key_here")
    MODEL_NAME = config.get("gemini", {}).get("model_name", "gemini-2.5-flash")
    model_name = "gemini-2.5-flash"
    llm = GeminiLLM(
        api_key=API_KEY,
        model=MODEL_NAME
    )
else:
    llm = LLMInference(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        quantization_config=quantization_config,  # 使用預設配置
    )

vectorstore = load_chroma_vectorstore(
    collection_name="llm_course",
    persist_directory=str(vector_store_path),
    embedding_function=embedding_model
)
# 向量庫重置
vectorstore.reset_collection()

#### PDF處理 ####

pdf_path = base_dir / "pdf_input" / "一口桃滋味-營養午餐亮點食譜.pdf"
docs = load_pdf(pdf_path = str(pdf_path))
for doc in docs[:3]:
    print(doc.page_content[:100], "...")
    print("Metadata:", doc.metadata)
    print("-" * 120)


docs = split_documents(documents=docs,
                  chunk_size=4096,
                  chunk_overlap=50)

if len(docs) == 0:
    print("沒有讀取到任何文件片段，請檢查PDF文件內容或分割參數。")
    exit(1)

vectorstore.add_documents(docs)
conversation_history = []
temperature = 0.4
max_tokens = 10240

system_prompt = f"""
你是一個專業的回答助理，你會收到使用者詢問的問題，以及使用者提供的相關文件內容。請根據這些文件內容回答問題。
如果文件內容無法回答使用者的問題，請直接回答「檔案中並未提及相關資訊。」，不需要回答其他內容。"
"""
conversation_history.append({"role": "system", "content": system_prompt})

while True:
    
    user_input = input("請輸入查詢問題 (或輸入 'exit' 退出): ")
    if user_input.lower() == 'exit':
        break
    conversation_history.append({"role": "user", "content": user_input})
    results = vectorstore.similarity_search_with_relevance_scores(
        user_input, 
        k=2,
    )
    document_string = ""
    for doc, score in results:
        # print(f"Score = {score}")
        # print("Document Content:", doc.page_content)
        # print("Metadata:", doc.metadata)
        # print("-" * 80)
        document_string += f"Score: {score}\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n\n"
    prompt = f"""
    
    使用者問題：\n\n{user_input}\n\n"
    
    相關文件內容：\n\n
    {document_string}
    
    """
    print("生成提示：", prompt)
    response = ""
    conversation_history.append({"role": "user", "content": prompt})
    
    print("模型：", end="", flush=True)
    
    for chunk in llm.generate(
        messages=conversation_history,
        temperature=temperature,
        max_new_tokens=max_tokens
    ):
        response += chunk
        print(chunk, end="", flush=True)
    print("\n")
    # 添加助手回應到歷史
    conversation_history.append({"role": "assistant", "content": response})

