from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import re
import hashlib


project_root = Path(__file__).resolve().parent

path_to_db = project_root / "chroma_db"


def get_file_hash(pdf_path):
    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def add_document(pdf_path):

    # 1. Calculate document hash
    file_hash = get_file_hash(pdf_path)

    # 2. Connect to EXISTING Chroma DB
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="University_RAG",
        embedding_function=embedding_model,
        persist_directory=path_to_db,
    )

    # 3. Check whether this document already exists
    existing = vector_store.get(
        where={"document_hash": file_hash},
        limit=1
    )

    if existing["ids"]:
        return {"status":"duplicate","chunks":0}

    # 4. Load the PDF
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # 5. Clean text
    for doc in documents:
        doc.page_content = re.sub(
            r"\s+",
            " ",
            doc.page_content
        ).strip()

    # 6. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1300,
        chunk_overlap=450,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    # 7. Add document hash to every chunk
    for chunk in chunks:
        chunk.metadata["document_hash"] = file_hash

    # 8. Add new chunks to Chroma
    vector_store.add_documents(chunks)

    return{"status":"added","chunks":len(chunks)}

if __name__ == "__main__":
    pdf_path = project_root / "knowledge_base" / "Document 23_ Student Council Constitution & Election Regulations.pdf"

    chunks_added = add_document(pdf_path)

    print(f"Successfully added {chunks_added} chunks.")