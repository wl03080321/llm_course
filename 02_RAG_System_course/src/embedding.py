from langchain_huggingface import HuggingFaceEmbeddings
import torch

def load_embedding_model(model_name, cache_folder="./cache"):
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name = str(model_name),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder = cache_folder  # Specify your cache directory
    )
    return embedding_model