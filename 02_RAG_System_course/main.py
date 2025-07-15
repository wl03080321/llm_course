# Documentation: https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
from langchain_core.documents import Document
from src.vectordatabase import load_chroma_vectorstore, check_vectorstore
from src.embedding import load_embedding_model
from pathlib import Path as path


base_dir = path(__file__).parent.parent
vector_store_path = base_dir / "vectorstore"
model_path = base_dir / "cache"

# https://huggingface.co/sentence-transformers
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 768
embedding_model = load_embedding_model(
    model_name=embedding_model_name, cache_folder=str(model_path)
)

vectorstore = load_chroma_vectorstore(
    collection_name="llm_course",
    persist_directory=str(vector_store_path),
    embedding_function=embedding_model,
)
# 重製向量庫
vectorstore.reset_collection()

#### 簡單文檔敘述 ####

# 定義文件內容
document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={
        "source": "tweet",
        "page": 1,
        "filename": "breakfast_tweet.pdf",
        "type": "breakfast",
    },
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={
        "source": "news",
        "page": 2,
        "filename": "weather_news.pdf",
        "type": "weather",
    },
)

ids = ["1", "2"]
documents = [document_1, document_2]

print("添加文檔至向量庫：", end="\n")
for doc in documents:
    print(f"文檔內容: {doc.page_content}...")
    print(f"元數據: {doc.metadata}")
    print("-" * 120, end="\n\n")

vectorstore.add_documents(documents, ids=ids)

# 檢查向量庫結果
print("向量庫內容：")
check_vectorstore(vectorstore=vectorstore)

print("更新ids=2文檔：")
update_document = Document(
    page_content="今天天氣真好，陽光明媚，氣溫適中。",
    metadata={"source": "news", "page": 3, "filename": "台灣.pdf", "type": "weather"},
)
vectorstore.update_document(document_id="2", document=update_document)
check_vectorstore(vectorstore=vectorstore)

print("刪除ids=1的文檔")
vectorstore.delete(ids=["1"])

check_vectorstore(vectorstore=vectorstore)
