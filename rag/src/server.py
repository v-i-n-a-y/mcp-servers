import argparse
from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def get_embedding_function(model="embeddinggemma"):
    embeddings = OllamaEmbeddings(model="embeddinggemma")
    return embeddings


mcp = FastMCP("RAG")

CHROMA_PATH = "../chroma"


@mcp.tool()
def ragquery_score(query: str, k: int):
    """Perform similarity search and return formatted RAG prompt context."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    result = db.similarity_search_with_score(query, k=k)

    # Format context into a readable, LLM-friendly block
    context_blocks = []
    for i, (doc, score) in enumerate(result):
        context_blocks.append(
            f"Document {i+1} (score={score:.3f}):\n{doc.page_content.strip()}"
        )

    context_text = "\n\n".join(context_blocks)

    # Build final RAG-style prompt text
    prompt = f"""You are an expert assistant with access to retrieved documents.
Use the information below to answer the user's query accurately.

Context:
{context_text}

Question:
{query}

If the answer is not found in the context, clearly say so.
"""

    return {
        "query": query,
        "k": k,
        "prompt": prompt,
        "documents": [
            {"score": score, "page_content": doc.page_content}
            for doc, score in result
        ],
    }


@mcp.tool()
def ragquery(query: str):
    """Simpler version without scores."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    result = db.similarity_search(query)

    context_text = "\n\n".join(
        f"Document {i+1}:\n{doc.page_content.strip()}" for i, doc in enumerate(result)
    )

    prompt = f"""You are an expert assistant with access to retrieved documents.

Context:
{context_text}

Question:
{query}
"""

    return {
        "query": query,
        "prompt": prompt,
        "documents": [doc.page_content for doc in result],
    }



def main():
    mcp.run()


if __name__ == "__main__":
    main()
