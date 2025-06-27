# Documentation: https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb import PersistentClient
from src.utils.pdf_process import load_pdf, split_text
from src.embedding import load_embedding_model
from pathlib import Path as path
from typing import List, Dict, Optional, Tuple, Any
import os


class RAGSystem:
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "llm_course",
                 persist_directory: Optional[str] = None,
                 cache_folder: Optional[str] = None):
        """
        初始化 RAG 系統
        
        Args:
            embedding_model_name: 嵌入模型名稱
            collection_name: 向量資料庫集合名稱
            persist_directory: 向量資料庫儲存目錄
            cache_folder: 模型快取資料夾
        """
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # 設定路徑
        self.base_dir = path(__file__).parent.parent.parent
        self.vector_store_path = persist_directory or str(self.base_dir / "vectorstore")
        self.model_path = cache_folder or str(self.base_dir / "cache")
        self.embedding_model = self._load_embedding_model()
        self.vectorstore = None
    
    def _load_embedding_model(self):
        """載入嵌入模型"""
        print(f"Loading embedding model: {self.embedding_model_name}")
        embedding_model = load_embedding_model(
            model_name=self.embedding_model_name, 
            cache_folder=self.model_path
        )
        print("Embedding model loaded successfully!")
        return embedding_model
    
    def _load_vectorstore(self):
        """載入向量資料庫"""
        print(f"Loading vector store from: {self.vector_store_path}")
        vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.vector_store_path,
            embedding_function=self.embedding_model
        )
        print("Vector store loaded successfully!")
        return vectorstore
    
    def _ensure_vectorstore_loaded(self):
        """確保向量資料庫已載入"""
        if self.vectorstore is None:
            print("Loading vector store on demand...")
            self.vectorstore = self._load_vectorstore()
        return self.vectorstore
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        將文件新增到向量資料庫
        
        Args:
            documents: 要新增的文件列表
            ids: 文件 ID 列表（可選）
            
        Returns:
            新增文件的 ID 列表
        """
        try:
            vectorstore = self._ensure_vectorstore_loaded()
            if ids:
                result_ids = vectorstore.add_documents(documents, ids=ids)
            else:
                result_ids = vectorstore.add_documents(documents)
            
            print(f"Successfully added {len(documents)} documents to vector store")
            return result_ids
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return []
    
    def similarity_search(self, query: str, k: int = 4, 
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        執行相似度搜尋
        
        Args:
            query: 查詢文字
            k: 返回的文件數量
            filter_dict: 過濾條件
            
        Returns:
            相似的文件列表
        """
        try:
            vectorstore = self._ensure_vectorstore_loaded()
            if filter_dict:
                results = vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                results = vectorstore.similarity_search(query, k=k)
            
            print(f"Found {len(results)} similar documents for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_scores(self, query: str, k: int = 4,
                                    filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        執行相似度搜尋並返回分數
        
        Args:
            query: 查詢文字
            k: 返回的文件數量
            filter_dict: 過濾條件
            
        Returns:
            (文件, 相似度分數) 的元組列表
        """
        try:
            vectorstore = self._ensure_vectorstore_loaded()
            if filter_dict:
                results = vectorstore.similarity_search_with_relevance_scores(
                    query, k=k, filter=filter_dict
                )
            else:
                results = vectorstore.similarity_search_with_relevance_scores(
                    query, k=k
                )
            
            print(f"Found {len(results)} documents with scores for query: '{query}'")
            for i, (doc, score) in enumerate(results):
                print(f"  {i+1}. [Score: {score:.4f}] {doc.page_content[:100]}...")
            
            return results
            
        except Exception as e:
            print(f"Error during similarity search with scores: {e}")
            return []
    
    def get_documents(self, filter_dict: Optional[Dict] = None, 
                     limit: Optional[int] = None) -> Dict[str, Any]:
        """
        從向量資料庫獲取文件
        
        Args:
            filter_dict: 過濾條件
            limit: 限制返回數量
            
        Returns:
            包含文件資訊的字典
        """
        try:
            vectorstore = self._ensure_vectorstore_loaded()
            if filter_dict:
                docs = vectorstore.get(where=filter_dict, limit=limit)
            else:
                docs = vectorstore.get(limit=limit)
            
            print(f"Retrieved {len(docs['ids'])} documents from vector store")
            return docs
            
        except Exception as e:
            print(f"Error getting documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}
    
    def process_pdf_file(self, pdf_path: str, chunk_size: int = 500, 
                        chunk_overlap: int = 50, add_to_vectorstore: bool = True) -> List[Document]:
        """
        處理 PDF 檔案
        
        Args:
            pdf_path: PDF 檔案路徑
            chunk_size: 文字分塊大小
            chunk_overlap: 分塊重疊大小
            add_to_vectorstore: 是否自動新增到向量資料庫
            
        Returns:
            處理後的文件列表
        """
        try:
            # 載入 PDF
            print(f"Loading PDF from: {pdf_path}")
            docs = load_pdf(pdf_path=pdf_path)
            print(f"Successfully loaded {len(docs)} pages from PDF")
            
            # 為每個文件添加檔案名稱到 metadata
            filename = os.path.basename(pdf_path)
            for doc in docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['filename'] = filename
            
            # 分割文字
            if chunk_size > 0:
                print(f"Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})")
                docs = split_text(
                    documents=docs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                # 確保分割後的文件也有檔案名稱
                for doc in docs:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['filename'] = filename
                print(f"Created {len(docs)} text chunks")
            
            # 自動新增到向量資料庫
            if add_to_vectorstore:
                self.add_documents(docs)
            
            return docs
            
        except Exception as e:
            print(f"Error processing PDF file: {e}")
            return []
    

    def delete_documents_by_filename(self, filename: str) -> Dict[str, Any]:
        """
        根據檔案名稱刪除所有相關的文件
        
        Args:
            filename: 要刪除的檔案名稱
            
        Returns:
            刪除結果
        """
        result = {
            "success": False,
            "message": "",
            "filename": filename,
            "deleted_count": 0
        }
        
        try:
            vectorstore = self._ensure_vectorstore_loaded()
            
            # 搜尋該檔案的所有文件
            docs = vectorstore.get(where={"filename": filename})
            
            if not docs["ids"]:
                result["message"] = f"沒有找到檔案 '{filename}' 的相關文件"
                return result
            
            # 刪除文件
            vectorstore.delete(ids=docs["ids"])
            
            result["success"] = True
            result["deleted_count"] = len(docs["ids"])
            result["message"] = f"成功刪除檔案 '{filename}' 的 {len(docs['ids'])} 個文件"
            
            print(f"Deleted {len(docs['ids'])} documents for file: {filename}")
            
        except Exception as e:
            error_msg = f"刪除檔案 '{filename}' 的文件時發生錯誤: {str(e)}"
            result["message"] = error_msg
            print(error_msg)
        
        return result    
    
    def get_all_filenames(self) -> List[str]:
        """
        獲取向量資料庫中所有檔案名稱
        
        Returns:
            檔案名稱列表
        """
        try:
            vectorstore = self._ensure_vectorstore_loaded()
            docs = vectorstore.get()
            
            # 從 metadata 中提取所有檔案名稱
            filenames = set()
            for metadata in docs.get("metadatas", []):
                if metadata and "filename" in metadata:
                    filenames.add(metadata["filename"])
            
            return sorted(list(filenames))
            
        except Exception as e:
            print(f"Error getting filenames: {e}")
            return []
    
    def get_file_document_count(self, filename: str) -> int:
        """
        獲取指定檔案的文件數量
        
        Args:
            filename: 檔案名稱
            
        Returns:
            文件數量
        """
        try:
            vectorstore = self._ensure_vectorstore_loaded()
            docs = vectorstore.get(where={"filename": filename})
            return len(docs.get("ids", []))
            
        except Exception as e:
            print(f"Error getting document count for file '{filename}': {e}")
            return 0
        
    def process_uploaded_files(self, uploaded_files: List[str], 
                              upload_folder: str,
                              chunk_size: int = 500, 
                              chunk_overlap: int = 50,
                              add_to_vectorstore: bool = True) -> Dict[str, Any]:
        """
        處理從 Gradio 上傳的檔案列表
        
        Args:
            uploaded_files: 上傳的檔案名稱列表
            upload_folder: 檔案上傳的資料夾路徑
            chunk_size: 文字分塊大小
            chunk_overlap: 分塊重疊大小
            add_to_vectorstore: 是否自動新增到向量資料庫
            
        Returns:
            處理結果的詳細資訊
        """
        results = {
            "processed_files": [],
            "failed_files": [],
            "total_documents": 0,
            "file_details": {}
        }
        
        for filename in uploaded_files:
            file_path = os.path.join(upload_folder, filename)
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                results["failed_files"].append({"filename": filename, "error": "File not found"})
                continue
            
            try:
                # 根據檔案副檔名決定處理方式
                file_extension = os.path.splitext(filename)[1].lower()
                
                if file_extension == ".pdf":
                    docs = self.process_pdf_file(
                        pdf_path=file_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        add_to_vectorstore=add_to_vectorstore
                    )
                    
                    if docs:
                        results["processed_files"].append(filename)
                        results["total_documents"] += len(docs)
                        results["file_details"][filename] = {
                            "type": "pdf",
                            "chunks": len(docs),
                            "status": "success"
                        }
                    else:
                        results["failed_files"].append({"filename": filename, "error": "PDF processing failed"})
                        
                else:
                    print(f"Unsupported file type: {file_extension} for file: {filename}")
                    results["failed_files"].append({"filename": filename, "error": f"Unsupported file type: {file_extension}"})
                    
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                results["failed_files"].append({"filename": filename, "error": str(e)})
        
        return results
    
    def create_document(self, content: str, metadata: Dict = None) -> Document:
        """
        建立文件物件
        
        Args:
            content: 文件內容
            metadata: 元資料
            
        Returns:
            Document 物件
        """
        return Document(
            page_content=content,
            metadata=metadata or {}
        )
    
    def query_and_retrieve(self, query: str, k: int = 4, 
                          return_scores: bool = True) -> List[Dict]:
        """
        查詢並檢索相關文件
        
        Args:
            query: 查詢文字
            k: 返回的文件數量
            return_scores: 是否返回相似度分數
            
        Returns:
            檢索結果列表
        """
        try:
            if return_scores:
                results = self.similarity_search_with_scores(query, k=k)
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in results
                ]
            else:
                results = self.similarity_search(query, k=k)
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in results
                ]
                
        except Exception as e:
            print(f"Error during query and retrieve: {e}")
            return []
    
    def get_vectorstore_info(self) -> Dict:
        """
        獲取向量資料庫資訊
        
        Returns:
            向量資料庫資訊字典
        """
        try:
            docs = self.get_documents()
            filenames = self.get_all_filenames()
            return {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "vector_store_path": self.vector_store_path,
                "total_files": len(filenames),  # 檔案數量
                "total_documents": len(docs["ids"]),  # 文件片段數量
                "filenames": filenames,
                "document_ids": docs["ids"][:10] if docs["ids"] else []  # 只顯示前10個ID
            }
        except Exception as e:
            print(f"Error getting vectorstore info: {e}")
            return {}
    

    
    def get_rag_response(self, user_query: str, k: int = 3) -> str:
        """
        獲取 RAG 回應 (檢索 + 生成)
        
        Args:
            user_query: 使用者查詢
            k: 檢索的文件數量
            
        Returns:
            基於檢索結果的回應文字
        """
        try:
            # 檢索相關文件
            retrieved_docs = self.query_and_retrieve(user_query, k=k, return_scores=True)
            
            if not retrieved_docs:
                return "抱歉，我沒有找到相關的資訊來回答您的問題。"
            
            # 整理檢索到的內容
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                context_parts.append(f"參考資料 {i+1} (相似度: {doc['score']:.3f}):\n{doc['content']}")
            
            context = "\n\n".join(context_parts)
            
            # 建構回應 (這裡可以接 LLM 生成，目前先返回檢索結果)
            response = f"根據我找到的相關資料：\n\n{context}"
            
            return response
            
        except Exception as e:
            print(f"Error getting RAG response: {e}")
            return "處理您的查詢時發生錯誤，請稍後再試。"
    
    def get_processing_status(self, uploaded_files: List[str], upload_folder: str) -> str:
        """
        獲取檔案處理狀態文字 (用於 Gradio 顯示)
        
        Args:
            uploaded_files: 上傳的檔案列表
            upload_folder: 檔案上傳資料夾
            
        Returns:
            狀態文字
        """
        if not uploaded_files:
            return "目前沒有檔案需要處理。"
        
        status_lines = [f"準備處理 {len(uploaded_files)} 個檔案："]
        
        for filename in uploaded_files:
            file_path = os.path.join(upload_folder, filename)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                file_ext = os.path.splitext(filename)[1]
                status_lines.append(f"  ✓ {filename} ({file_size:.1f} KB, {file_ext})")
            else:
                status_lines.append(f"  ✗ {filename} (檔案不存在)")
        
        return "\n".join(status_lines)
