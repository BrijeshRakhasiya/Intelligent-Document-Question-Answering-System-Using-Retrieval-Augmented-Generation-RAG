import streamlit as st
from pathlib import Path
import os
import sqlite3
import hashlib
from PyPDF2 import PdfReader
from sqlalchemy import create_engine
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

# Page setup
st.set_page_config(
    page_title="AI Chat Assistant: SQL DB & PDF Q&A", 
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 AI Chat Assistant: SQL Database & PDF Documents")

# Initialize session state
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "SQL Database Chat"
if "sql_messages" not in st.session_state:
    st.session_state.sql_messages = [{"role": "assistant", "content": "How can I help you with your database?"}]
if "pdf_chat_history" not in st.session_state:
    st.session_state.pdf_chat_history = []
if "pdf_vector_store" not in st.session_state:
    st.session_state.pdf_vector_store = None
if "processed_files_hash" not in st.session_state:
    st.session_state.processed_files_hash = None
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Sidebar configuration
st.sidebar.title("Configuration")

# API Key Setup
if not st.session_state.api_key_set:
    st.sidebar.info("🔑 Enter your Groq API Key to get started")
    st.sidebar.markdown("Get your free API key from: https://console.groq.com/keys")
    
    groq_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", placeholder="gsk_...")
    
    if st.sidebar.button("Set API Key", type="primary") and groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.session_state.api_key_set = True
        st.sidebar.success("✅ API key set successfully!")
        st.rerun()
    
    if groq_key:
        st.sidebar.info("👆 Click 'Set API Key' button to continue")
    
    st.stop()

# Mode selection
mode = st.sidebar.radio(
    "Choose Chat Mode:",
    ["SQL Database Chat", "PDF Documents Q&A"],
    key="mode_selector"
)

st.session_state.current_mode = mode

# Constants for SQL Database
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# SQL Database Configuration
if st.session_state.current_mode == "SQL Database Chat":
    st.sidebar.subheader("Database Configuration")
    
    radio_opt = ["Use SQLite 3 Database - student.db", "Connect to your MySQL Database"]
    selected_opt = st.sidebar.radio("Choose the DB you want to chat with", options=radio_opt)

    if selected_opt == radio_opt[1]:
        db_uri = MYSQL
        
        # Check if MySQL connector is available
        try:
            import mysql.connector
            mysql_available = True
        except ImportError:
            mysql_available = False
            st.sidebar.warning("⚠️ MySQL connector not installed. Please install it using: `pip install mysql-connector-python`")
        
        mysql_host = st.sidebar.text_input("MySQL Host", value="localhost", placeholder="localhost or IP address")
        mysql_user = st.sidebar.text_input("MySQL User", value="root", placeholder="MySQL username")
        mysql_password = st.sidebar.text_input("MySQL Password", type="password", placeholder="MySQL password")
        mysql_db = st.sidebar.text_input("MySQL Database", placeholder="Database name")
        mysql_port = st.sidebar.number_input("MySQL Port", value=3306, min_value=1, max_value=65535)
    else:
        db_uri = LOCALDB

    # LLM setup for SQL
    @st.cache_resource
    def get_sql_llm():
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            streaming=True
        )

    # DB configuration with better error handling
    @st.cache_resource(ttl="2h")
    def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None, mysql_port=3306):
        if db_uri == LOCALDB:
            try:
                dbfilepath = (Path(__file__).parent / "student.db").absolute()
                if not dbfilepath.exists():
                    # Create a demo database if it doesn't exist
                    st.info("📁 Creating a demo SQLite database...")
                    try:
                        conn = sqlite3.connect(dbfilepath)
                        cursor = conn.cursor()
                        
                        # Create a sample students table
                        cursor.execute('''
                            CREATE TABLE IF NOT EXISTS students (
                                id INTEGER PRIMARY KEY,
                                name TEXT NOT NULL,
                                age INTEGER,
                                grade TEXT,
                                email TEXT
                            )
                        ''')
                        
                        # Insert sample data
                        sample_data = [
                            (1, 'John Smith', 20, 'A', 'john.smith@email.com'),
                            (2, 'Jane Doe', 22, 'B', 'jane.doe@email.com'),
                            (3, 'Mike Johnson', 21, 'A', 'mike.johnson@email.com'),
                            (4, 'Sarah Wilson', 19, 'C', 'sarah.wilson@email.com'),
                            (5, 'Tom Brown', 23, 'B', 'tom.brown@email.com')
                        ]
                        
                        cursor.executemany('''
                            INSERT OR IGNORE INTO students (id, name, age, grade, email)
                            VALUES (?, ?, ?, ?, ?)
                        ''', sample_data)
                        
                        conn.commit()
                        conn.close()
                        st.success("✅ Demo database created successfully!")
                    except Exception as e:
                        st.error(f"Error creating demo database: {str(e)}")
                        return None
                
                creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
                db_engine = create_engine("sqlite:///", creator=creator)
                return SQLDatabase(db_engine)
            except Exception as e:
                st.error(f"Error connecting to SQLite database: {str(e)}")
                return None
                
        elif db_uri == MYSQL:
            if not all([mysql_host, mysql_user, mysql_password, mysql_db]):
                st.error("Please provide all MySQL connection details.")
                return None
            
            # Check if MySQL connector is available
            try:
                import mysql.connector
            except ImportError:
                st.error("""
                ❌ MySQL connector not found. Please install it using:
                ```
                pip install mysql-connector-python
                ```
                """)
                return None
            
            try:
                # Try different MySQL connection strings
                connection_strings = [
                    f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}",
                    f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
                ]
                
                for conn_str in connection_strings:
                    try:
                        db_engine = create_engine(conn_str)
                        # Test connection
                        with db_engine.connect() as conn:
                            pass
                        st.success("✅ MySQL connection successful!")
                        return SQLDatabase(db_engine)
                    except Exception as e:
                        continue
                
                st.error("❌ Failed to connect to MySQL. Please check your connection details.")
                return None
                
            except Exception as e:
                st.error(f"MySQL connection error: {str(e)}")
                return None

    # SQL Agent setup
    @st.cache_resource
    def get_sql_agent(_db):
        if _db is None:
            return None
            
        llm = get_sql_llm()
        toolkit = SQLDatabaseToolkit(db=_db, llm=llm)
        
        return create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

    # Load DB and agent
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db, mysql_port)
    else:
        db = configure_db(db_uri)
        
    agent = get_sql_agent(db) if db else None

    # SQL Chat Interface
    st.header("🦜 Chat with SQL Database")
    
    # Database information
    if db_uri == LOCALDB:
        st.info("🔗 Connected to: SQLite Database (student.db)")
    elif db_uri == MYSQL:
        if db:
            st.info(f"🔗 Connected to: MySQL Database ({mysql_db})")
        else:
            st.info("🔗 MySQL: Please configure connection details")
    
    if db_uri == MYSQL and not all([mysql_host, mysql_user, mysql_password, mysql_db]):
        st.info("Please configure your MySQL connection in the sidebar.")
    elif agent is None:
        st.error("Failed to initialize database agent. Please check your configuration.")
    else:
        # Clear chat history button
        if st.sidebar.button("Clear SQL Chat History"):
            st.session_state.sql_messages = [{"role": "assistant", "content": "How can I help you with your database?"}]
            st.rerun()

        # Display sample questions
        with st.expander("💡 Sample questions you can ask"):
            st.markdown("""
            - "Show me all students"
            - "How many students are there?"
            - "Which students have grade A?"
            - "What's the average age of students?"
            - "Show students older than 20"
            """)

        # Display chat messages
        for msg in st.session_state.sql_messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat input
        user_query = st.chat_input("Ask anything from the database...")

        if user_query:
            st.session_state.sql_messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                try:
                    streamlit_callback = StreamlitCallbackHandler(st.container())
                    response = agent.run(user_query, callbacks=[streamlit_callback])
                except Exception as e:
                    response = f"⚠️ An error occurred: {str(e)}"
                st.session_state.sql_messages.append({"role": "assistant", "content": response})
                st.write(response)

# PDF Q&A Configuration
else:
    st.header("📚 Ask Questions about Your PDF Documents")
    
    # PDF processing functions with progress indicators
    @st.cache_data
    def get_pdf_text(pdf_files):
        text = ""
        pdf_names = []
        total_files = len(pdf_files)
        
        if total_files == 0:
            return text, pdf_names
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf in enumerate(pdf_files):
            pdf_names.append(pdf.name)
            status_text.text(f"📖 Reading {pdf.name}...")
            pdf_reader = PdfReader(pdf)
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                text += page_text
                # Update progress for pages within current file
                current_progress = (i + (page_num + 1) / total_pages) / total_files
                progress_bar.progress(current_progress)
                
        progress_bar.progress(1.0)
        status_text.empty()
        return text, pdf_names

    @st.cache_data
    def get_text_chunks(text):
        if not text.strip():
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return [chunk for chunk in chunks if chunk.strip()]  # Filter empty chunks

    @st.cache_resource
    def get_embeddings_model():
        try:
            # Use a simpler, more reliable embedding model
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            return embeddings
        except Exception as e:
            st.error(f"Error loading embeddings model: {str(e)}")
            return None

    def create_vector_store(text_chunks, embeddings):
        if not text_chunks:
            st.error("No text chunks available for processing.")
            return None
        
        progress_text = st.empty()
        progress_text.text("Creating vector store...")
        progress_bar = st.progress(0)
        
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = min(20, len(text_chunks))
            vector_store = None
            
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    vector_store.add_texts(batch)
                
                progress = min((i + batch_size) / len(text_chunks), 1.0)
                progress_bar.progress(progress)
                progress_text.text(f"Processing chunks: {min(i + batch_size, len(text_chunks))}/{len(text_chunks)}")
            
            progress_bar.progress(1.0)
            progress_text.empty()
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            progress_text.empty()
            return None

    @st.cache_resource
    def get_conversational_chain():
        if not os.getenv("GROQ_API_KEY"):
            st.error("API key is required")
            return None
            
        prompt_template = """You are a helpful assistant. Answer the question based on the provided context. If the answer is not in the context, say "I cannot find this information in the provided documents."

Context: {context}
Question: {question}
Please provide a helpful answer:"""
        
        try:
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1,
                max_tokens=512
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

    # PDF Chat Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.pdf_chat_history:
            for message in st.session_state.pdf_chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            if not st.session_state.pdf_vector_store:
                st.error("Please upload and process documents first.")
            else:
                with st.spinner("🔍 Searching through your documents..."):
                    try:
                        search_status = st.empty()
                        search_status.text("🔍 Finding relevant information...")
                        
                        # Perform similarity search
                        docs = st.session_state.pdf_vector_store.similarity_search(user_question, k=2)
                        
                        search_status.text("🤔 Generating answer...")
                        
                        chain = get_conversational_chain()
                        
                        if chain is None:
                            st.error("Please check your API configuration")
                        else:
                            response = chain({"input_documents": docs, "question": user_question})
                            
                            st.session_state.pdf_chat_history.append({"role": "user", "content": user_query})
                            st.session_state.pdf_chat_history.append({"role": "assistant", "content": response["output_text"]})
                            search_status.empty()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        if "401" in str(e) or "invalid_api_key" in str(e):
                            st.error("Invalid API key. Please check your Groq API key.")

    with col2:
        st.subheader("📄 Upload Documents")
        pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, 
                                   help="Upload one or more PDF files to analyze")
        
        if pdf_files:
            current_hash = get_file_hash(pdf_files)
            
            if current_hash != st.session_state.processed_files_hash:
                st.info(f"📄 {len(pdf_files)} document(s) ready for processing")
                
                if st.button("Process Documents", type="primary", key="process_btn"):
                    # Initialize embeddings first
                    with st.spinner("🔄 Loading embeddings model..."):
                        embeddings = get_embeddings_model()
                    
                    if embeddings is None:
                        st.error("❌ Failed to load embeddings model. Please check your internet connection and try again.")
                    else:
                        # Process documents
                        processing_container = st.container()
                        with processing_container:
                            st.info("📖 Extracting text from PDFs...")
                            raw_text, pdf_names = get_pdf_text(pdf_files)
                            
                            if not raw_text.strip():
                                st.error("❌ No text could be extracted from the PDFs. Please ensure the PDFs contain selectable text (not scanned images).")
                            else:
                                st.info(f"✅ Extracted {len(raw_text)} characters from {len(pdf_names)} documents")
                                
                                st.info("✂️ Splitting text into chunks...")
                                text_chunks = get_text_chunks(raw_text)
                                
                                if not text_chunks:
                                    st.error("❌ No text chunks created. The PDFs might be empty or contain only images.")
                                else:
                                    st.info(f"✅ Created {len(text_chunks)} text chunks")
                                    
                                    st.info("🔧 Creating search index...")
                                    vector_store = create_vector_store(text_chunks, embeddings)
                                    
                                    if vector_store is not None:
                                        st.session_state.pdf_vector_store = vector_store
                                        st.session_state.processed_files_hash = current_hash
                                        st.session_state.documents_processed = True
                                        st.session_state.pdf_chat_history = []
                                        
                                        st.success(f"✅ Successfully processed {len(pdf_names)} documents!")
                                        st.balloons()
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to create vector store. Please try again with smaller documents.")
            else:
                st.success("✅ Documents already processed!")
                if st.session_state.pdf_vector_store:
                    num_chunks = len(st.session_state.pdf_vector_store.index_to_docstore_id)
                    st.info(f"📊 Search index contains {num_chunks} text chunks")
        
        # Document processing status
        if st.session_state.pdf_vector_store:
            st.success("✅ Documents are ready for questioning!")
        
        # Clear chat button
        if st.session_state.pdf_chat_history:
            if st.button("Clear Chat History", key="clear_chat"):
                st.session_state.pdf_chat_history = []
                st.rerun()

        # Troubleshooting tips
        with st.expander("ℹ️ Troubleshooting Tips"):
            st.markdown("""
            **Common Issues and Solutions:**
            
            - **Slow processing**: Start with 1-2 small PDFs first
            - **No text extracted**: Ensure PDFs contain selectable text (not scanned images)
            - **Model loading errors**: Check your internet connection
            - **Memory issues**: Try with smaller PDFs or fewer documents
            
            **For best results:**
            - Use PDFs with text content (not image-based scans)
            - Start with documents under 10 pages each
            - Ensure good internet connection for model downloads
            """)

# Installation requirements in sidebar
with st.sidebar.expander("📋 Installation Requirements"):
    st.markdown("""
    **Required packages:**
    ```bash
    pip install streamlit langchain langchain-community langchain-groq faiss-cpu sentence-transformers PyPDF2 sqlalchemy python-dotenv
    ```
    
    **For MySQL support (optional):**
    ```bash
    pip install mysql-connector-python
    ```
    """)

# Footer information
st.sidebar.markdown("---")
st.sidebar.info(
    "**Quick Tips:**\n"
    "- SQL Mode: Works with SQLite (default) or MySQL\n"
    "- PDF Mode: Upload text-based PDFs for best results\n" 
    "- Start small with 1-2 documents to test"
)