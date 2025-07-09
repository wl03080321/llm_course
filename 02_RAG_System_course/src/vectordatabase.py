from typing import Optional, List, Union
from langchain_chroma import Chroma

def load_chroma_vectorstore(
        collection_name=None,
        persist_directory="./chroma_db",
        embedding_function=None
    ):
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def check_vectorstore(vectorstore: Chroma,
                      ids:Optional[Union[str, List[str]]] = None,
                      where:Optional[dict] = None,
                      limit:Optional[int] = 10,
                      ):
    docs = vectorstore.get(
        ids=ids,
        where=where,
        limit=limit
    )
    for key, value in docs.items():
        print(f"{key}: {value}")
    print("\n")
    return docs
