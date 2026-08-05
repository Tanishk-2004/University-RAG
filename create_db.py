from langchain_community.document_loaders import PyPDFLoader  # Used for loading the pdf
from pathlib import Path 
from langchain_text_splitters import RecursiveCharacterTextSplitter  # used for chunking
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
import sys

project_root = Path(__file__).resolve().parent

knowledge_base=project_root/ "knowledge_base"



pdf_files=knowledge_base.glob("*.pdf")     # this will store all the files ending with .pdf into the pdf_files variable 



documents=[]  # TO add all the pages 

# # now we use for loop to deal with pdf one one at a time and also create empty documents list so that later we can add all the pdfs in one place 
# #  we convert pdf into the string coz the pyPDFloader expects string 


for pdf in pdf_files:
    loader = PyPDFLoader(str(pdf))
    loaded_pages = loader.load()
    documents.extend(loaded_pages)
    
# ## To replace the \n coz its causing each word to go to new line and make our text look bad we use this
import re
for doc in documents:
    doc.page_content=re.sub(r"\s+"," ",doc.page_content).strip() 


text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1300,
    chunk_overlap=450,
    length_function=len,
    separators=["\n\n","\n"," ",""]  
)

chunks = text_splitter.split_documents(documents) # now the dcoument is splitted into smaller smaller chunks 

print(chunks[0].page_content)


#Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

path_to_db = project_root / "chroma_dbb"

print(path_to_db)

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="University_RAG",
    persist_directory=path_to_db,
)
