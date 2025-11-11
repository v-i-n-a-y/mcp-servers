from langchain_ollama import OllamaEmbeddings


def get_embedding_function(model="embeddinggemma"):
    embeddings = OllamaEmbeddings(model="embeddinggemma")
    return embeddings
