"""
🎬 FANDOM CHATBOT - Exercise 1: Dive into RAG!
================================================
Workshop: Bringing Your Data to Life: Dive into RAG
By: Rizwan Rahim, CTO of DataVine

This script builds a simple RAG (Retrieval-Augmented Generation) pipeline
that lets you chat with any PDF document using a LOCAL LLM (no API keys needed!)

The 6-Step RAG Journey:
    1. Environment Setup (Install dependencies inside venv)
    2. Document Loading (PyPDFLoader)
    3. Text Splitting (RecursiveCharacterTextSplitter)
    4. Vector Storage (FAISS + HuggingFace Embeddings)
    5. RAG Chain (create_retrieval_chain)
    6. Interactive Query Loop

Think of this as teaching an AI to become an instant expert on whatever PDF you give it!
"""

import os
import sys

# ============================================================================
# 🎨 STEP 1: ENVIRONMENT SETUP
# ============================================================================
# The magic ingredients are in requirements.txt
# Run: pip install -r requirements.txt
# 
# We're using LOCAL tools - no expensive API keys!
# - Ollama: Runs LLMs on your computer (like having ChatGPT at home)
# - HuggingFace Embeddings: Converts text to numbers (locally!)
# - FAISS: Lightning-fast similarity search

print("\n" + "="*60)
print("🎬 FANDOM CHATBOT - Dive into RAG!")
print("="*60)


# ============================================================================
# 📚 STEP 2: DOCUMENT LOADING
# ============================================================================
# We use PyPDFLoader to read PDF files
# It's like opening a book and reading each page into memory

from langchain_community.document_loaders import PyPDFLoader

def load_document(file_path: str):
    """
    Load a PDF document and return its pages.
    
    Think of this as hiring a librarian who reads your PDF
    and hands you each page as a separate document.
    """
    print(f"\n📖 Loading document: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        print("   Make sure your PDF is in the same folder as this script!")
        sys.exit(1)
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"✅ Loaded {len(pages)} pages from the PDF")
    return pages


# ============================================================================
# ✂️ STEP 3: TEXT SPLITTING
# ============================================================================
# Why do we split text? Great question!
# 
# LLMs have a "context window" - like a reading window that can only see
# a limited amount of text at once (usually 4K-128K tokens).
# 
# If your PDF is 100 pages, we can't send it all at once!
# So we split it into smaller, digestible chunks.
#
# ANALOGY: Imagine reading a book through a magnifying glass that only
# shows 1 paragraph at a time. We need to find the RIGHT paragraph!
#
# Parameters:
# - chunk_size=1000: Each chunk is ~1000 characters (about 1 paragraph)
# - chunk_overlap=100: Chunks overlap by 100 chars (so we don't cut sentences mid-way)

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    Split documents into smaller chunks for processing.
    
    The 'Recursive' part means it tries to split smartly:
    1. First by paragraphs (\\n\\n)
    2. Then by sentences (.)
    3. Then by words
    4. Finally by characters
    
    This keeps semantic meaning intact!
    """
    print("\n✂️  Splitting document into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # ~1 paragraph per chunk
        chunk_overlap=100,    # 10% overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks from the document")
    print(f"   Average chunk size: ~{sum(len(c.page_content) for c in chunks)//len(chunks)} characters")
    
    return chunks


# ============================================================================
# 🧭 STEP 4: VECTOR STORAGE (The GPS for Meaning!)
# ============================================================================
# This is where the magic happens! We convert text to NUMBERS (vectors).
#
# ANALOGY: Think of embeddings as GPS coordinates for MEANING.
# - "happy" and "joyful" have similar coordinates (close together)
# - "happy" and "motorcycle" have different coordinates (far apart)
#
# When you ask a question, we find chunks with similar "coordinates"!
#
# We're using:
# - HuggingFace Embeddings: Runs locally, no API needed!
# - FAISS: Facebook's library for super-fast similarity search

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """
    Convert text chunks into vectors and store them in FAISS.
    
    This is like creating a searchable map where each location
    (chunk) has GPS coordinates (embeddings) based on its meaning.
    """
    print("\n🧭 Creating vector store (this might take a minute on first run)...")
    print("   Downloading embedding model if needed...")
    
    # Using a small, fast, and effective model
    # 'all-MiniLM-L6-v2' is only ~90MB and works great!
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a compatible GPU!
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # FAISS creates an index for lightning-fast similarity search
    # It's like building a GPS system for your document
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    print("✅ Vector store created successfully!")
    print(f"   {len(chunks)} chunks are now searchable by meaning")
    
    return vector_store


# ============================================================================
# 🔗 STEP 5: THE RAG CHAIN
# ============================================================================
# Now we connect everything together:
# 1. Retriever: Finds relevant chunks from the vector store
# 2. LLM: Generates answers based on those chunks
# 3. Chain: Orchestrates the whole flow
#
# We use Ollama for the LLM - it runs models like Llama 3.2 on YOUR computer!

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

def create_rag_chain(vector_store, model_name: str = "llama3.2"):
    """
    Create the RAG chain that connects retrieval to generation.
    
    The flow:
    Question → Retriever (finds relevant chunks) → LLM (generates answer)
    
    The LLM is instructed to ONLY use information from the retrieved chunks.
    This prevents hallucination and keeps answers grounded in your PDF!
    """
    print(f"\n🔗 Creating RAG chain with Ollama ({model_name})...")
    
    # Check if Ollama is running
    try:
        llm = OllamaLLM(
            model=model_name,
            temperature=0.3,  # Lower = more focused answers
        )
        # Quick test to see if Ollama is responsive
        llm.invoke("Hi")
    except Exception as e:
        print(f"\n❌ Error connecting to Ollama: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Make sure Ollama is installed: curl -fsSL https://ollama.com/install.sh | sh")
        print("   2. Start Ollama: ollama serve")
        print(f"   3. Pull the model: ollama pull {model_name}")
        sys.exit(1)
    
    # The prompt template - this is crucial!
    # We tell the LLM to ONLY use the provided context (retrieved chunks)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer cannot be found in the context, say "I don't have enough information in the document to answer that."

Do not make up information. Stick to what's in the context.

Context from the document:
{context}

Question: {input}

Answer (be concise and cite specific details from the context):
""")
    
    # Create the document chain (combines retrieved docs with the prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create a retriever from the vector store
    # k=4 means we retrieve the 4 most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Combine everything into the final RAG chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    print("✅ RAG chain ready!")
    print(f"   Using model: {model_name}")
    print("   Retrieval: Top 4 most relevant chunks per question")
    
    return rag_chain


# ============================================================================
# 💬 STEP 6: INTERACTIVE QUERY LOOP
# ============================================================================
# The fun part! Ask questions and get answers grounded in your PDF.

def chat_loop(rag_chain):
    """
    Interactive loop where users can ask questions about their document.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "="*60)
    print("💬 CHAT MODE - Ask questions about your document!")
    print("="*60)
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("🙋 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Fandom Chatbot! See you next time!")
                break
            
            print("\n🤔 Thinking...")
            
            # Invoke the RAG chain
            response = rag_chain.invoke({"input": question})
            
            # Extract the answer
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")
            
            print(f"\n🤖 Bot: {answer}\n")
            print("-" * 40 + "\n")
            
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
    Main function that orchestrates the entire RAG pipeline.
    """
    print("\n🚀 Starting the Fandom Chatbot...")
    
    # ========================================
    # 📄 CONFIGURE YOUR PDF AND MODEL HERE
    # ========================================
    file_path = "/home/ba7man/RAG/Exercise1/Titanic_Alternate_Ending.pdf"
    model_name = "llama3.2"
    # ========================================
    
    print(f"\n📄 Using PDF: {file_path}")
    print(f"🧠 Using model: {model_name}")
    
    # Run the pipeline
    try:
        # Step 2: Load the document
        documents = load_document(file_path)
        
        # Step 3: Split into chunks
        chunks = split_documents(documents)
        
        # Step 4: Create vector store
        vector_store = create_vector_store(chunks)
        
        # Step 5: Create RAG chain
        rag_chain = create_rag_chain(vector_store, model_name)
        
        # Step 6: Start chatting!
        chat_loop(rag_chain)
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("\n🔧 Common fixes:")
        print("   1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Make sure Ollama is running: ollama serve")
        print(f"   3. Make sure the model is pulled: ollama pull {model_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
