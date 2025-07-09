from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=50, separators=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    split_docs = splitter.split_documents(documents)
    return split_docs

def split_text(texts, chunk_size=500, chunk_overlap=50, separators=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    split_docs = splitter.split_text(texts)
    return split_docs