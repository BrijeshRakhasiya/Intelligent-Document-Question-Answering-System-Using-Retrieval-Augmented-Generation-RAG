import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import hashlib

load_dotenv()

st.set_page_config(page_title="Multi-PDF RAG with Groq", page_icon="📚", layout="wide")
st.title("Ask Your PDFs with Groq's Llama 3")

if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if not st.session_state.api_key_set:
    st.info("🔑 Enter your Groq API Key to get started")
    st.markdown("Get your free API key from: https://console.groq.com/keys")
    
    groq_key = st.text_input("Enter your Groq API Key:", type="password", placeholder="gsk_...")
    
    if st.button("Set API Key", type="primary") and groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.session_state.api_key_set = True
        st.success("✅ API key set successfully!")
        st.rerun()
    
    if groq_key:
        st.info("👆 Click 'Set API Key' button to continue")
    
    st.stop()

@st.cache_data
def get_pdf_text(pdf_files):
    text = ""
    pdf_names = []
    for pdf in pdf_files:
        pdf_names.append(pdf.name)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text, pdf_names

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(text_chunks, embeddings):
    cleaned_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]
    if not cleaned_chunks:
        return None
    return FAISS.from_texts(cleaned_chunks, embedding=embeddings)

@st.cache_resource
def get_conversational_chain():
    if not os.getenv("GROQ_API_KEY"):
        st.error("API key is required")
        return None
        
    prompt_template = """Answer the question based only on the provided context. If the answer is not in the context, say "I cannot find this information in the provided documents."

Context: {context}
Question: {question}
Answer:"""
    
    try:
        llm = ChatGroq(
            model_name="llama3-70b-8192", 
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Failed to initialize Groq: {str(e)}")
        return None

def get_file_hash(pdf_files):
    content = ""
    for pdf in pdf_files:
        content += pdf.name + str(pdf.size)
    return hashlib.md5(content.encode()).hexdigest()

def main():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files_hash" not in st.session_state:
        st.session_state.processed_files_hash = None

    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        user_question = st.chat_input("Ask a question about your documents:")
        
        if user_question and st.session_state.vector_store:
            with st.spinner("Searching..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(user_question, k=3)
                    chain = get_conversational_chain()
                    
                    if chain is None:
                        st.error("Please check your API configuration")
                        return
                        
                    response = chain({"input_documents": docs, "question": user_question})
                    
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": response["output_text"]})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
                    if "401" in str(e) or "invalid_api_key" in str(e):
                        st.error("Invalid API key. Please check your Groq API key.")

    with col2:
        st.subheader("📄 Upload Documents")
        pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        
        if pdf_files:
            current_hash = get_file_hash(pdf_files)
            
            if current_hash != st.session_state.processed_files_hash:
                if st.button("Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        raw_text, pdf_names = get_pdf_text(pdf_files)
                        text_chunks = get_text_chunks(raw_text)
                        embeddings = get_embeddings_model()
                        
                        st.session_state.vector_store = create_vector_store(text_chunks, embeddings)
                        st.session_state.processed_files_hash = current_hash
                        st.session_state.chat_history = []
                        
                        st.success(f"Processed {len(pdf_names)} documents with {len(text_chunks)} chunks")
                        st.rerun()
            else:
                st.success("Documents already processed!")
        
        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()