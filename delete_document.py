from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib
from pathlib import Path 


project_root = Path(__file__).resolve().parent

path_to_db = project_root / "chroma_db"


def get_file_hash(pdf_path):
    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def delete_document(pdf_path):
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

    # 3. Find all chunks belonging to this document
    existing = vector_store.get(
        where={"document_hash": file_hash}
    )
    
    if not existing["ids"]:
        return {"status": "not_found", "chunks": 0}

    # 4. Delete only this document's chunks
    vector_store.delete(ids=existing["ids"])

    # 5. Verify that the chunks were deleted
    remaining = vector_store.get(
        where={"document_hash": file_hash}
    )

    if remaining["ids"]:
        return {"status": "delete_failed", "chunks": 0}

    # 6. Delete the PDF from the knowledge base
    pdf_path = Path(pdf_path)

    if pdf_path.exists():
        pdf_path.unlink()

    return {"status": "deleted", "chunks": len(existing["ids"])}


if __name__ == "__main__":
    pdf_path = (
        project_root
        / "knowledge_base"
        / "Document 23_ Student Council Constitution & Election Regulations.pdf"
    )

    result = delete_document(pdf_path)

    if result["status"] == "deleted":
        print(
            f"Successfully deleted document and "
            f"{result['chunks']} chunks."
        )
    elif result["status"] == "not_found":
        print("Document was not found in the ChromaDB.")
    else:
        print("Document deletion failed.")
