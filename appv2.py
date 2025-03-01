# First, let's update the imports to ensure we have all necessary components
import streamlit as st
from streamlit_chat import message
import chromadb
import os
import time
import pandas as pd
import numpy as np
from llama_index.core import (
    VectorStoreIndex, 
    Document,
    StorageContext,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import io
import sys
import fitz  # PyMuPDF for PDF handling
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import uuid
import traceback
import glob
import json
from typing import List, Dict, Any, Optional, Union, Tuple

# Read the API key from env_variables.txt
def load_api_key(file_path="env_variables.txt"):
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split("=")
            if key == "OPENAI_API_KEY":  # Adjust based on your key name
                return value
    return None

# Load API key
api_key = load_api_key()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing for different file types"""
    
    @staticmethod
    def read_pdf(file_content: bytes) -> str:
        try:
            # Using PyMuPDF (fitz) for PDF processing
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise Exception(f"Failed to process PDF: {str(e)}")

    @staticmethod
    def read_txt(file_content: bytes) -> str:
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            for encoding in encodings:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise Exception("Could not decode text file with any supported encoding")
        except Exception as e:
            logger.error(f"Error reading TXT: {str(e)}")
            raise Exception(f"Failed to process TXT: {str(e)}")

class RAGChatbot:
    def __init__(self):
        self.setup_embeddings()
        self.setup_openai()
        self.initialize_session_state()
        self.doc_processor = DocumentProcessor()

    def setup_embeddings(self):
        """Initialize embedding models with error handling"""
        try:
            self.embed_model = OpenAIEmbedding(
                model="text-embedding-ada-002",
                api_key=api_key,
                request_timeout=50.0
            )
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 1024
            Settings.chunk_overlap = 200
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {str(e)}")
            raise
        
    def setup_openai(self):
        """Initialize OpenAI LLM with a persistent system prompt and optional custom user prompt."""
        max_retries = 3
        retry_delay = 2

        # Persistent System Prompt (contains context rules and essential data)
        system_prompt = (
            "You are an AI assistant designed to strictly follow provided context and data rules. "
            "Use only relevant information, adhere to data constraints, and maintain clarity in responses. "
            "If external context is provided, integrate it appropriately."
        )

        for attempt in range(max_retries):
            try:
                # Optional Custom Prompt from Streamlit UI
                custom_prompt = st.session_state.get("custom_user_prompt", "")

                # Combine system prompt with optional custom prompt
                combined_prompt = f"{system_prompt}\n\n{custom_prompt}" if custom_prompt else system_prompt

                # Initialize OpenAI LLM with the combined prompt
                self.openai_llm = OpenAI(
                    model="gpt-4o",  # Use "gpt-3.5-turbo" if needed
                    temperature=0.3,
                    system_prompt=combined_prompt,
                    api_key=api_key,
                    request_timeout=50.0
                )

                break

            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to initialize OpenAI after {max_retries} attempts: {str(e)}")
                    raise
                time.sleep(retry_delay)

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        session_vars = {
            "messages": [],
            "documents": None,
            "vector_index": None,
            "tree_index": None,
            "list_index": None,
            "keyword_index": None,
            "processing_status": None,
            "error_message": None,
            "pending_delete": None,
        }
        
        for var, default in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default

    def process_file(self, uploaded_file) -> Optional[Document]:
        """Process individual uploaded file with comprehensive error handling"""
        try:
            file_content = uploaded_file.read()
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = self.doc_processor.read_pdf(file_content)
            elif file_extension == 'txt':
                text = self.doc_processor.read_txt(file_content)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Clean and validate the extracted text
            text = self.clean_text(text)
            if not text or len(text.strip()) < 10:  # Minimum content validation
                raise ValueError("Extracted text is too short or empty")
                
            return Document(
                text=text,
                metadata={
                    "filename": uploaded_file.name,
                    "file_type": file_extension,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not isinstance(text, str):
            return ""
            
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.,!?-]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Remove very long words (likely garbage)
        text = ' '.join(word for word in text.split() if len(word) < 45)
        return text

    def create_indices(self, documents: List[Document]):
        """Create various indices with progress tracking and error handling"""
        try:
            st.session_state.documents = documents
            total_steps = 1
            progress_bar = st.progress(0)
            
            # Vector Index
            with st.spinner("Creating Vector Index... "):
                # Initialize ChromaDB client (persistent storage)
                chroma_client = chromadb.PersistentClient(path="./chroma_db")

                # Set up LlamaIndex with ChromaDB
                chroma_collection = chroma_client.get_or_create_collection("rag_vectors")
                vector_store = ChromaVectorStore(chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # Build and persist the index
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
                index.storage_context.persist("./chroma_index")
                
                st.session_state.vector_index = index

                print("Documents stored in ChromaDB!")
            progress_bar.progress(1/total_steps)
            
            st.success("All indices created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating indices: {str(e)}")
            st.error(f"Failed to create indices: {str(e)}")
            raise

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate normalized semantic similarity between two texts (0-1 scale)"""
        try:
            if not text1 or not text2:
                return 0.0
                
            # Enhanced text normalization
            text1 = self.clean_text(text1)[:512]  # Limit input size
            text2 = self.clean_text(text2)[:512]
            
            embeddings1 = self.embed_model.encode(text1)
            embeddings2 = self.embed_model.encode(text2)
            similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
            return max(0.0, min(1.0, similarity))  # Clamp to 0-1 range
        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")
            return 0.0

    def query_index(self, query: str, similarity_top_k: int = 3, 
                   temperature: float = 0.1, num_output: int = 1024) -> Dict[str, Any]:
        """Query the specified index with comprehensive error handling and metrics"""
        start_time = time.time()
        try:
            # Input validation
            if not query or len(query.strip()) < 2:
                raise ValueError("Query is too short or empty")
            
            # Create query engine with parameters
            query_engine = st.session_state.vector_index.as_query_engine(
                similarity_top_k=similarity_top_k,
                temperature=temperature,
                num_output=num_output,
                llm=self.openai_llm,
            )

            # Execute query
            response = query_engine.query(query)
            response_text = str(response)
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            similarity_score = self.calculate_similarity(query, response_text)
            performance_metric = similarity_score / elapsed_time if elapsed_time > 0 else 0

            return {
                "response": response_text,
                "elapsed_time": elapsed_time,
                "similarity_score": similarity_score,
                "performance_metric": performance_metric,
                "source_nodes": getattr(response, 'source_nodes', [])
            }

        except Exception as e:
            logger.error(f"Error during query execution: {str(e)}")
            raise
        
class DocumentManager:
    """Handles document loading and metadata retrieval from ChromaDB"""
    
    def __init__(self, db_path="./chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection("rag_vectors")
    
    def load_documents(self):
        """Fetches stored document metadata from ChromaDB and removes duplicates"""
        try:
            all_documents = self.collection.get(include=['metadatas'])
            metadata_list = all_documents['metadatas'] if 'metadatas' in all_documents else []
            
            # Deduplicate by filename
            unique_docs = {}
            for doc in metadata_list:
                filename = doc.get('filename', 'Unknown File')
                if filename not in unique_docs:
                    unique_docs[filename] = doc
            
            return list(unique_docs.values())
        except Exception as e:
            logger.error(f"Error loading documents from ChromaDB: {str(e)}")
            return []
        
    def queue_deletion(self, filename: str) -> bool:
        """Adds a document to the deletion queue for processing by the external script."""
        try:
            # Check if file is already in queue
            existing_queue = []
            if os.path.exists("delete_queue.txt"):
                with open("delete_queue.txt", "r") as f:
                    existing_queue = [line.strip() for line in f.readlines() if line.strip()]
                    
                if filename in existing_queue:
                    logger.info(f"File '{filename}' is already queued for deletion")
                    return True
            
            # Add to queue
            with open("delete_queue.txt", "a") as f:
                f.write(f"{filename}\n")
                
            # Also add to deletion log
            with open("deletion_queue_log.txt", "a") as log:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"[{timestamp}] Queued '{filename}' for deletion\n")
                
            logger.info(f"Document '{filename}' queued for deletion")
            return True
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Failed to queue deletion for '{filename}': {str(e)}")
            logger.error(f"Error details: {error_details}")
            
            # Try to write to an error file in case logging is failing
            try:
                with open("deletion_error_log.txt", "a") as err_log:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    err_log.write(f"[{timestamp}] Failed to queue '{filename}': {str(e)}\n")
                    err_log.write(f"Details:\n{error_details}\n\n")
            except:
                pass  # Last resort - if we can't write to file, just continue
                
            return False
            
    def perform_safe_deletion(self, filename):
        """Queues a document for deletion and shows feedback to the user."""
        if self.queue_deletion(filename):
            st.sidebar.success(f"'{filename}' marked for deletion")
            st.sidebar.info("Use the 'Process Deletion Queue' button to run deletions safely")
        else:
            st.sidebar.error(f"Failed to queue '{filename}' for deletion")
            
    def check_deletion_queue(self):
        """Checks for pending deletions and provides a button to run the external deletion script."""
        if not os.path.exists("delete_queue.txt"):
            return
            
        # Read the queue file
        with open("delete_queue.txt", "r") as f:
            files_to_delete = [line.strip() for line in f.readlines() if line.strip()]
            
        if not files_to_delete:
            if os.path.exists("delete_queue.txt"):
                os.remove("delete_queue.txt")
            return
            
        # Show status and button
        st.sidebar.info(f"Found {len(files_to_delete)} documents queued for deletion")
        
        # Check if deletion is already running
        deletion_running = os.path.exists("deletion_status.json")
        
        if deletion_running:
            try:
                # Read status from the status file
                with open("deletion_status.json", "r") as f:
                    status = json.load(f)
                    
                # Show progress
                processed = status.get("processed", 0)
                total = status.get("total", len(files_to_delete))
                succeeded = len(status.get("succeeded", []))
                failed = len(status.get("failed", []))
                
                st.sidebar.warning(f"Deletion in progress: {processed}/{total} processed")
                st.sidebar.success(f"Successfully deleted: {succeeded}")
                
                if failed > 0:
                    st.sidebar.error(f"Failed to delete: {failed}")
                    
                # Show refresh button
                if st.sidebar.button("Refresh Status"):
                    pass  # No need to do anything, will refresh on next render
                    
            except Exception as e:
                st.sidebar.error(f"Error reading deletion status: {str(e)}")
                
        else:
            # Show button to start deletion
            if st.sidebar.button("Process Deletion Queue"):
                try:
                    # Run the external script as a background process
                    import subprocess
                    
                    # Use Popen to run the script without blocking
                    process = subprocess.Popen(
                        ["python", "delete_docs.py"], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    st.sidebar.info("Started deletion process in background. Refresh the page to see progress.")
                    
                except Exception as e:
                    st.sidebar.error(f"Error starting deletion process: {str(e)}")
                    logger.error(f"Error starting deletion process: {str(e)}")
                    
        # If status file exists but process is done (all files processed)
        if deletion_running:
            try:
                with open("deletion_status.json", "r") as f:
                    status = json.load(f)
                
                processed = status.get("processed", 0)
                total = status.get("total", 0)
                
                if processed >= total and total > 0:
                    if st.sidebar.button("Clear Deletion Status"):
                        # Remove the status file
                        os.remove("deletion_status.json")
                        # Refresh
                        st.experimental_rerun()
            except:
                pass

def log_query_metrics(query: str, metrics: dict):
    """Log query metrics for historical analysis"""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        **metrics
    }
    
    try:
        # Append to history file
        with open("query_history.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log metrics: {str(e)}")

def process_deletion_queue_safely():
    """Process document deletion queue from a previous session before initializing ChromaDB.
    This standalone function runs before any ChromaDB operations to prevent app crashes."""
    
    # Skip if no deletion queue exists
    if not os.path.exists("delete_queue.txt"):
        return
        
    try:
        # Read the deletion queue
        with open("delete_queue.txt", "r") as f:
            files_to_delete = [line.strip() for line in f.readlines() if line.strip()]
            
        if not files_to_delete:
            os.remove("delete_queue.txt")
            return
            
        # Write to simple log file
        with open("deletion_attempt.txt", "w") as log:
            log.write(f"Attempting to delete {len(files_to_delete)} files at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for filename in files_to_delete:
                log.write(f"- {filename}\n")
        
        # We don't actually process deletions here - just return file count
        # Instead, we'll keep the queue for the DocumentManager to handle more carefully
        return (0, len(files_to_delete))
            
    except Exception as e:
        # Just log to a file since we're before any UI is available
        with open("deletion_error.txt", "w") as f:
            f.write(f"Error reading deletion queue: {str(e)}")
        return (0, 0)

def process_files(uploaded_files, chatbot):
    """Process uploaded files and create indices."""
    try:
        with st.spinner("Processing documents..."):
            documents = []
            for file in uploaded_files:
                try:
                    doc = chatbot.process_file(file)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue

            if documents:
                chatbot.create_indices(documents)
                st.sidebar.success(f"Successfully processed {len(documents)} documents")
            else:
                st.error("No documents were successfully processed")

    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        logger.error(f"Document processing error: {str(e)}")

def main():
    # Process deletion queue before initializing ChromaDB
    process_deletion_queue_safely()
    
    # Initialize chatbot
    try:
        chatbot = RAGChatbot()
        doc_manager = DocumentManager()

        # Load index if exists
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("rag_vectors")
        vector_store = ChromaVectorStore(chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)

        st.session_state.vector_index = index
        logger.info("Loaded index from ChromaDB.")
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return

    # Initialize session state variables if they don't exist
    if "pending_delete" not in st.session_state:
        st.session_state.pending_delete = None
    
    # Main UI
    st.title("AI Doc Genie Demo Instance")
    
    # Process deletion queue if needed - this will add UI to sidebar
    doc_manager.check_deletion_queue()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📤Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if uploaded_files:
            # Process documents when upload button is clicked
            if st.button("Process Documents", key="process_button"):
                process_files(uploaded_files, chatbot)

        # Document Management
        st.header("📁Document Management")
        document_list = doc_manager.load_documents()
        
        if document_list:
            # Display document list
            st.subheader("Available Documents")
            for doc in document_list:
                filename = doc.get('filename', 'Unknown File')
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {filename}")
                with col2:
                    # Only show delete button if not in confirmation mode
                    if st.button("🗑️", key=f"delete_{filename}"):
                        # Mark for deletion 
                        doc_manager.perform_safe_deletion(filename)
        else:
            st.info("No documents found in ChromaDB.")

        # Add advanced parameters
        with st.expander("Advanced Parameters"):
            similarity_top_k = st.slider("Top K Results", 1, 10, 3)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
            num_output = st.slider("Max Output Tokens", 256, 2048, 1024)
        
        # System Prompt is always sent (no user control)
        st.title("AI Custom Instructions")

        # Optional Custom Prompt
        custom_prompt = st.text_area(
            "Add any additional instructions for the AI (optional):",
            placeholder="e.g., Answer in a casual tone or focus on financial aspects."
        )

        # Update Custom Prompt in Session
        if st.button("Apply Custom Instructions"):
            st.session_state["custom_user_prompt"] = custom_prompt
            st.success("Custom instructions applied!")

        # Display current prompts
        st.write("### Persistent System Prompt:")
        st.code("You are an AI assistant designed to strictly follow provided context and data rules...")

        if "custom_user_prompt" in st.session_state:
            st.write("### Current Custom Prompt:")
            st.code(st.session_state["custom_user_prompt"])
        else:
            st.write("### No Custom Prompt Applied")

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Query input and processing
    if prompt := st.chat_input("What would you like to know?"):
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.vector_index:
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        result = chatbot.query_index(
                            prompt,
                            similarity_top_k,
                            temperature,
                            num_output
                        )
                        
                        st.markdown(result["response"])
                        
                        with st.expander("Query Metrics"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("⏱️ Response Time", f"{result['elapsed_time']:.2f}s")
                            with col2:
                                st.metric("🎯 Similarity Score", f"{result['similarity_score']:.2f}")
                            with col3:
                                st.metric("⚡ Performance", f"{result['performance_metric']:.2f}")
                                
                            if result.get("source_nodes"):
                                st.write("📚 Source Documents:")
                                for idx, node in enumerate(result["source_nodes"]):
                                    filename = node.metadata.get('filename', 'Unknown')
                                    score = getattr(node, 'score', 'N/A')
                                    st.write(f"- **Source {idx+1}**: {filename} (Score: {score:.4f})")
                            
                        log_query_metrics(prompt, result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"]
                        })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.error("Please upload and process documents first!")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Built with Streamlit & LlamaIndex")

if __name__ == "__main__":
    st.set_page_config(
        page_title=" AI Doc Genie",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    main()
    # with open('./creds.yaml') as file:
    #     config = yaml.load(file, Loader=SafeLoader)

    # # Pre-hashing all plain text passwords once
    # # stauth.Hasher.hash_passwords(config['credentials'])

    # authenticator = stauth.Authenticate(
    #     config['credentials'],
    #     config['cookie']['name'],
    #     config['cookie']['key'],
    #     config['cookie']['expiry_days']
    # )

    # authenticator.login()
    
    # # Authentication status handling
    # if st.session_state['authentication_status']:
    #     main()
    #     authenticator.logout("Logout", "sidebar")

    # elif st.session_state['authentication_status'] is False:
    #     st.error("Username or password is incorrect")

    # elif st.session_state['authentication_status'] is None:
    #     st.warning("Please enter your username and password")
