from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pathlib import Path 
import time
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate


project_root = Path(__file__).resolve().parent

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
persist_directory=project_root/"chroma_db"
# use it to get info from the existing database.

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name="University_RAG"
)


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":5,
        "fetch_k":15,
        "lambda_mult":1
    }
)


env_path = project_root/".env"

load_dotenv(env_path) # to use the env file containing the API key 

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for Hyderabad Institute of Technology.

Answer the user's question using ONLY the provided context.

The question may contain multiple parts.
Answer every part that can be answered from the context.

If the context contains multiple policies or multiple values for the same concept, 
explain which policy each value belongs to instead of assuming they refer to the same situation.




Only if none of the required information exists in the context, reply exactly:
"I don't know based on the provided documents."

Do not make up information.

Keep the answer concise and accurate.
                                          
Context:
{context}

Question:
{question}







Answer:
""")


def ask_question(question):

    response = multi_query_retriever.invoke(question)

    context = "\n=========\n".join(
        doc.page_content for doc in response
    )


    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    answer = llm.invoke(final_prompt)

   

    return {"answer":answer.content,"source_documents":response}

