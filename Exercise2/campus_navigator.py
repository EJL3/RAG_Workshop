"""
🏫 SMART CAMPUS NAVIGATOR - Exercise 2: Multi-Document RAG with Persistence
=============================================================================
Workshop: Bringing Your Data to Life: Dive into RAG
By: Rizwan Rahim, CTO of DataVine

This script builds a PRODUCTION-READY RAG pipeline that:
    ✅ Handles MULTIPLE documents (not just one PDF)
    ✅ PERSISTS the vector database to disk (no re-processing!)
    ✅ Uses ChromaDB for scalable, local vector storage

The Key Upgrades from Exercise 1:
    1. DirectoryLoader: Mass-ingest entire folders of documents
    2. ChromaDB: Save your vector store to disk permanently
    3. Smart Loading: Only process documents once!

Think of this as building a local search engine for your campus!
"""

import os
import sys

# ============================================================================
# ⚙️ CONFIGURATION
# ============================================================================
# Edit these paths to match your setup

DATA_FOLDER = "/home/ba7man/RAG/Exercise2/Campus_data"  # Folder with your .txt/.pdf files
CHROMA_DB_PATH = "/home/ba7man/RAG/Exercise2/chroma_campus_db"  # Where to save the vector database
COLLECTION_NAME = "campus_documents"  # Name for the ChromaDB collection
MODEL_NAME = "llama3.2"  # Ollama model to use

# ============================================================================

print("\n" + "="*65)
print("🏫 SMART CAMPUS NAVIGATOR - Multi-Document RAG")
print("="*65)


# ============================================================================
# 📚 STEP 1: DIRECTORY LOADING (Mass Ingestion!)
# ============================================================================
# Instead of loading one PDF at a time, we load an ENTIRE FOLDER.
# DirectoryLoader walks through your folder and loads every matching file.
#
# This is how production systems work - you don't manually load each document!
#
# We use:
# - DirectoryLoader: Loads all files matching a pattern
# - TextLoader: Handles .txt files
# - PyPDFLoader: Handles .pdf files (if you add any)

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

def load_documents_from_folder(folder_path: str):
    """
    Load ALL documents from a folder.
    
    Think of this as a librarian who grabs every book from a shelf
    and hands them to you in one stack.
    
    We handle both .txt and .pdf files for flexibility.
    """
    print(f"\n📂 Loading documents from: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        print("   Create the folder and add your .txt or .pdf files!")
        sys.exit(1)
    
    all_documents = []
    
    # Load all .txt files
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if txt_files:
        print(f"   📄 Found {len(txt_files)} text files: {txt_files}")
        txt_loader = DirectoryLoader(
            folder_path,
            glob="**/*.txt",           # Match all .txt files
            loader_cls=TextLoader,      # Use TextLoader for .txt
            loader_kwargs={'encoding': 'utf-8'}
        )
        all_documents.extend(txt_loader.load())
    
    # Load all .pdf files (if any exist)
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if pdf_files:
        print(f"   📕 Found {len(pdf_files)} PDF files: {pdf_files}")
        pdf_loader = DirectoryLoader(
            folder_path,
            glob="**/*.pdf",           # Match all .pdf files
            loader_cls=PyPDFLoader     # Use PyPDFLoader for .pdf
        )
        all_documents.extend(pdf_loader.load())
    
    if not all_documents:
        print("❌ No .txt or .pdf files found in the folder!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(all_documents)} document chunks from folder")
    
    return all_documents


# ============================================================================
# ✂️ STEP 2: TEXT SPLITTING (Same as Exercise 1)
# ============================================================================
# We still need to split documents into chunks for the LLM's context window.

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    Split documents into smaller chunks.
    Same concept as Exercise 1 - keeping semantic meaning intact!
    """
    print("\n✂️  Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Smaller chunks for more precise retrieval
        chunk_overlap=50,     # 10% overlap
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks from all documents")
    
    return chunks


# ============================================================================
# 💾 STEP 3: PERSISTENT VECTOR STORE (The Game Changer!)
# ============================================================================
# This is the BIG upgrade from Exercise 1!
#
# In Exercise 1 (FAISS):
#   - Vector store lives in RAM (memory)
#   - Lost when you close the program
#   - Must re-process documents every time
#
# In Exercise 2 (ChromaDB):
#   - Vector store saved to disk (persist_directory)
#   - Survives program restarts
#   - Process once, use forever!
#
# This is a PRODUCTION-READY habit. Real companies don't re-process
# their entire knowledge base every time they restart their servers!

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_embeddings():
    """
    Initialize the embedding model (same as Exercise 1).
    We use HuggingFace embeddings that run locally.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def create_or_load_vector_store(chunks=None):
    """
    Create a NEW vector store, or LOAD an existing one.
    
    This is the magic of persistent storage:
    - First run: Process documents and save to disk
    - Future runs: Load instantly from disk!
    
    It's like the difference between:
    - Cooking a meal from scratch every time (Exercise 1)
    - Meal prepping once and reheating (Exercise 2) 🍱
    """
    embeddings = get_embeddings()
    
    # Check if database already exists
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print("\n💾 Found existing vector database!")
        print(f"   Loading from: {CHROMA_DB_PATH}")
        
        # Load the existing database
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        # Get collection info
        collection = vector_store._collection
        count = collection.count()
        print(f"✅ Loaded {count} vectors from disk (instant!)")
        
        return vector_store
    
    # No existing database - create a new one
    if chunks is None:
        print("❌ No existing database and no documents provided!")
        sys.exit(1)
    
    print("\n🧭 Creating NEW vector store...")
    print(f"   This will be saved to: {CHROMA_DB_PATH}")
    print("   (Future runs will load instantly!)")
    
    # Create the vector store with persistence
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH  # ⭐ This saves to disk!
    )
    
    print(f"✅ Created and persisted {len(chunks)} vectors to disk!")
    
    return vector_store


# ============================================================================
# 🔗 STEP 4: THE RAG CHAIN (Context-Aware Retrieval)
# ============================================================================
# Same pattern as Exercise 1, but now we're querying MULTIPLE documents!
# The retriever will find the most relevant chunks from ANY document.

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

def create_rag_chain(vector_store):
    """
    Create the RAG chain for multi-document Q&A.
    
    The beauty is that the user doesn't need to know WHICH document
    contains the answer - the retriever finds it automatically!
    """
    print(f"\n🔗 Creating RAG chain with Ollama ({MODEL_NAME})...")
    
    # Initialize Ollama LLM
    try:
        llm = OllamaLLM(
            model=MODEL_NAME,
            temperature=0.3,
        )
        llm.invoke("Hi")  # Quick test
    except Exception as e:
        print(f"\n❌ Error connecting to Ollama: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Start Ollama: ollama serve")
        print(f"   2. Pull the model: ollama pull {MODEL_NAME}")
        sys.exit(1)
    
    # Campus-specific prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful Campus Navigator assistant. Answer questions based ONLY on the provided context.

The context contains information from multiple campus documents including:
- Mess menu and food schedules
- Hostel rules and regulations  
- Campus events and activities

If the answer is not in the context, say "I don't have that information in my campus database."

Context from campus documents:
{context}

Student's Question: {input}

Helpful Answer:
""")
    
    # Create the chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retriever with more results for multi-doc
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Top 5 chunks from ANY document
    )
    
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    print("✅ RAG chain ready!")
    print(f"   Retrieval: Top 5 chunks across ALL documents")
    
    return rag_chain


# ============================================================================
# 💬 STEP 5: INTERACTIVE QUERY LOOP
# ============================================================================

def chat_loop(rag_chain):
    """
    Interactive loop for campus queries.
    """
    print("\n" + "="*65)
    print("💬 CAMPUS NAVIGATOR - Ask me anything about campus life!")
    print("="*65)
    print("Try: 'What's for lunch on Thursday?' or 'When is Tech Fest?'")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("🎓 Student: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Good luck with your studies! See you around campus!")
                break
            
            print("\n🔍 Searching campus database...")
            
            response = rag_chain.invoke({"input": question})
            answer = response.get("answer", "Sorry, I couldn't find that information.")
            
            # Show which documents were used (helpful for debugging)
            source_docs = response.get("context", [])
            sources = set()
            for doc in source_docs:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.add(os.path.basename(doc.metadata['source']))
            
            print(f"\n🤖 Navigator: {answer}")
            if sources:
                print(f"\n   📚 Sources: {', '.join(sources)}")
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Try rephrasing your question.\n")


# ============================================================================
# 🚀 MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function - orchestrates the production RAG pipeline.
    """
    print("\n🚀 Initializing Smart Campus Navigator...")
    print(f"   Data folder: {DATA_FOLDER}")
    print(f"   Database: {CHROMA_DB_PATH}")
    
    # Check if we need to process documents
    need_processing = not (os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH))
    
    if need_processing:
        print("\n📋 First run detected - processing documents...")
        
        # ====================================================================
        # 🧪 PERSISTENCE DEMO:
        # ====================================================================
        # After running this script ONCE, try commenting out the two lines
        # below (Step 1 and Step 2). Then run the script again.
        # 
        # You'll see the AI STILL knows the answers! Why?
        # Because the vector database is SAVED to disk in chroma_campus_db/
        # 
        # This is the power of PERSISTENT storage - process once, use forever!
        # In production, you don't re-process your knowledge base every restart.
        # ====================================================================
        
        # Step 1: Load documents from folder
        # 👇 COMMENT THIS LINE AFTER FIRST RUN TO TEST PERSISTENCE 👇
        documents = load_documents_from_folder(DATA_FOLDER)
        
        # Step 2: Split into chunks
        # 👇 COMMENT THIS LINE AFTER FIRST RUN TO TEST PERSISTENCE 👇
        chunks = split_documents(documents)
        
        # Step 3: Create and persist vector store
        vector_store = create_or_load_vector_store(chunks)
    else:
        # ====================================================================
        # ⚡ THIS IS THE MAGIC OF PERSISTENCE!
        # ====================================================================
        # When chroma_campus_db/ already exists, we skip ALL document processing
        # and load the pre-computed vectors INSTANTLY from disk.
        # 
        # Try it: Run once, then run again - notice how fast the second run is!
        # ====================================================================
        print("\n⚡ Existing database found - loading instantly!")
        
        # Just load the existing vector store
        vector_store = create_or_load_vector_store()
    
    # Step 4: Create RAG chain
    rag_chain = create_rag_chain(vector_store)
    
    # Step 5: Start chatting!
    chat_loop(rag_chain)


if __name__ == "__main__":
    main()
