from src.vectordatabase import load_chroma_vectorstore, check_vectorstore
from src.embedding import load_embedding_model
from src.pdf_process import load_pdf, split_text, split_documents
from pathlib import Path as path

base_dir = path(__file__).parent.parent
vector_store_path = base_dir / "vectorstore"
model_path = base_dir / "cache"
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embedding_model = load_embedding_model(
    model_name=embedding_model_name,
    cache_folder=str(model_path)
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

while True:
    query = input("請輸入查詢問題 (或輸入 'exit' 退出): ")
    if query.lower() == 'exit':
        break

    results = vectorstore.similarity_search_with_relevance_scores(
        query, 
        k=5,
    )

    for doc, score in results:
        print(f"Score = {score}")
        print("Document Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 80)

