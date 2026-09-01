from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import re
import hashlib

project_root = Path(__file__).resolve().parent

knowledge_base = project_root / "knowledge_base"

pdf_files = knowledge_base.glob("*.pdf")

documents = []


def get_file_hash(pdf_path):
    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


for pdf in pdf_files:

    file_hash = get_file_hash(pdf)

    loader = PyPDFLoader(str(pdf))
    loaded_pages = loader.load()

    for doc in loaded_pages:
        doc.metadata["document_hash"] = file_hash
        doc.metadata["scope"] = "global"

    documents.extend(loaded_pages)


import re

for doc in documents:
    doc.page_content = re.sub(
        r"\s+",
        " ",
        doc.page_content
    ).strip()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1300,
    chunk_overlap=450,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

path_to_db = project_root / "chroma_db"

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="University_RAG",
    persist_directory=path_to_db,
)