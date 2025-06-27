# Documentation: https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.utils.pdf_process import load_pdf, split_text
from src.embedding import load_embedding_model
from pathlib import Path as path

base_dir = path(__file__).parent.parent
vector_store_path = base_dir / "vectorstore"
model_path = base_dir / "cache"

def load_chroma_vectorstore(collection_name=None,
                            persist_directory="./chroma_db",
                            embedding_function=None):
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function
        )
    return vectorstore

# https://huggingface.co/sentence-transformers
# embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
# embedding_model = load_embedding_model(model_name=embedding_model_name, cache_folder=str(model_path))

# vectorstore = load_chroma_vectorstore(
#     collection_name="llm_course",
#     persist_directory=str(vector_store_path),
#     embedding_function=embedding_model
#     )

## 定義文件內容
# document_1 = Document(
#     page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
#     metadata={"source": "tweet", "page": 1, "filename" : "breakfast_tweet.pdf", "type": "breakfast"},
# )

# document_2 = Document(
#     page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
#     metadata={"source": "news", "page": 2, "filename" : "weather_news.pdf","type": "weather"},
# )
# ids = ["1", "2"]
# documents = [document_1, document_2]
# print(documents)
# vectorstore.add_documents(documents, ids=ids)

# # 檢查向量庫結果
# docs = vectorstore.get(where={"type": "breakfast"})
# print(type(docs))
# print(docs['ids'])
# print(docs['embeddings'])
# print(docs['metadatas'])
# print(docs['documents'])

# query = "What did I have for breakfast?"

# results = vectorstore.similarity_search_with_relevance_scores(
#     query, 
#     k=2,
# )

# for doc, score in results:
#     print(f"* [SIM={score}] {doc.page_content} [{doc.metadata}]")


#### PDF處理 ####

pdf_path = base_dir / "pdf_input" / "KN-V24AT食譜集.pdf"
docs = load_pdf(pdf_path = str(pdf_path))
print("讀取完成")
print(f"讀取 {len(docs)} 個文件")
print(f"{docs}")

docs = split_text(documents=docs,
                  chunk_size=500,
                  chunk_overlap=50)

print(f"讀取 {len(docs)} 個片段")
for id, item in enumerate(docs):
    print(f"片段:{id}")
    print(f"頁數:{item.metadata}")
    print(f"內容:\n{item.page_content[:100]}...\n")
    print("="*50)


# vectorstore.add_documents(documents)