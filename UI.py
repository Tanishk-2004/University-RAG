import streamlit as st
from app import ask_question

# ==========================================
# DEFINE PAGE 1: ABOUT
# ==========================================
def page_about():
    st.title("📚 My RAG Project")

    st.subheader("What the project does")
    st.write("This project allows you to ask questions about your documents. It reads the PDFs, finds the most relevant information, and uses AI to give you a clear and accurate answer.")

    st.subheader("Tools & Technologies Used")
    st.write("- **Python**: Core programming language")
    st.write("- **Streamlit**: For building this simple user interface")
    st.write("- **LangChain**: To connect the AI with the documents")
    st.write("- **ChromaDB**: To store the document data (Vector Database)")
    st.write("- **OpenAI / HuggingFace**: The brain (LLM) that answers the questions")

    st.subheader("High-Level Architecture")
    st.write("**PDFs → Chunking → Embeddings → ChromaDB → Retrieval → LLM → Answer**")


# ==========================================
# DEFINE PAGE 2: LIVE DEMO
# ==========================================
def page_demo():
    st.header("🤖 Live RAG Demo")
    st.write("Ask the AI any question based on the provided documents.")

    user_question = st.text_input("What is your question?", placeholder="e.g., What is the minimum attendance required?")

    if st.button("Ask"):
        if user_question:
            with st.spinner("Searching documents and generating answer..."):
                answer = ask_question(user_question)
            st.success(answer)
            
        else:
            st.warning("Please type a question first!")


# ==========================================
# DEFINE PAGE 3: CONTACT
# ==========================================
def page_contact():
    st.header("📬 Contact")
    st.write("**Built by Tanishk**")
    st.write("- [LinkedIn](https://linkedin.com/in/yourprofile)")
    st.write("- [GitHub](https://github.com/yourgithub)")
    st.write("- 📧 Email: your.email@example.com")


# ==========================================
# SETUP NAVIGATION MENU
# ==========================================
about_page = st.Page(page_about, title="About Project", icon="📚")
demo_page = st.Page(page_demo, title="Live Demo", icon="🤖")
contact_page = st.Page(page_contact, title="Contact", icon="📬")

# Connect pages to navigation bar
pg = st.navigation([about_page, demo_page, contact_page])
pg.run()