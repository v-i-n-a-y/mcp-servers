import argparse
import os
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor,  ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_core.documents import Document
from langchain_chroma import Chroma

from embedding import get_embedding_function


cwd = os.getcwd()
DATA_PATH = os.path.join(cwd, "data")
CHROMA_PATH = "chroma"


def load_documents(max_workers: int = 16) -> list[Document]:
    """Loads all PDF files concurrently using a thread pool."""
    documents = []
    lock = threading.Lock()

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]

    def process_file(filename: str):
        try:
            file_path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(
                file_path,
                mode="page",
                images_inner_format="markdown-img",
                images_parser=TesseractBlobParser(),
            )
            docs = loader.load()
            with lock:
                documents.extend(docs)
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f) for f in pdf_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Loading PDFs"):
            pass

    print(f"Finished loading {len(documents)} documents.")
    return documents


def _split_single_doc(doc: Document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    title = doc.metadata.get("title") or doc.metadata.get("source") or "Untitled Document"
    print(f"Splitting document: {os.path.basename(title)}")
    return text_splitter.split_documents([doc])


def split_documents(documents: list[Document], max_workers: int = 8):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunks in executor.map(_split_single_doc, documents):
            results.extend(chunks)
    return results


def add_to_db(chunks: list[Document]):
    
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        batch_size = 1000
        batches = [
            (new_chunks[i : i + batch_size], new_chunk_ids[i : i + batch_size])
            for i in range(0, len(new_chunks), batch_size)
        ]

        def add_batch(batch_chunks, batch_ids):
            db.add_documents(batch_chunks, ids=batch_ids)
        
        threads = []
        for batch_chunks, batch_ids in batches:
            thread = threading.Thread(target=add_batch, args=(batch_chunks, batch_ids))
            threads.append(thread)
            thread.start()

        
        for thread in threads:
            thread.join()
    else:
        print("All documents are already added.")


def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0
    lock = threading.Lock()

    def process_chunk(chunk):
        nonlocal last_page_id, current_chunk_index
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        with lock:
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            chunk.metadata["id"] = chunk_id
            last_page_id = current_page_id

    threads = []
    for chunk in chunks:
        thread = threading.Thread(target=process_chunk, args=(chunk,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_db(chunks)


if __name__ == "__main__":
    main()
