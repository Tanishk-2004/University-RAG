from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import re


project_root = Path(__file__).resolve().parent

path_to_db = project_root / "chroma_db"


def add_document(pdf_path):

    # 1. Load the PDF
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # 2. Clean text
    for doc in documents:
        doc.page_content=re.sub(r"\s+"," ",doc.page_content).strip() 

    # 3. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1300,
        chunk_overlap=450,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    # 4. Embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Connect to EXISTING Chroma DB
    vector_store = Chroma(
        collection_name="University_RAG",
        embedding_function=embedding_model,
        persist_directory=path_to_db,
    )

    # 6. Add new chunks
    vector_store.add_documents(chunks)

    return len(chunks)

if __name__ == "__main__":
    pdf_path = project_root / "knowledge_base" / "Document 21_ Campus Health Services & Medical Emergency Protocols.pdf"

    chunks_added = add_document(pdf_path)

    print(f"Successfully added {chunks_added} chunks.")