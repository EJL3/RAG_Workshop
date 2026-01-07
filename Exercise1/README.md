# 🎬 Fandom Chatbot - Exercise 1: Dive into RAG!

> **Workshop:** Bringing Your Data to Life: Dive into RAG  
> **By:** Rizwan Rahim, CTO of DataVine

Welcome! In this exercise, you'll build a **Retrieval-Augmented Generation (RAG)** pipeline that can answer questions about ANY PDF you give it. Think of it as teaching an AI to become an expert on your favorite movie, anime, or book!

---

## 🚀 Quick Start

### Step 1: Install Ollama (Your Local AI Brain)

Ollama lets you run powerful AI models on your own computer - no API keys, no costs!

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Then pull a model (llama3.2 is fast and smart!)
ollama pull llama3.2
```

> **Pro Tip:** `llama3.2` is lightweight (~2GB). For better quality, try `mistral` or `llama3.1`

### Step 2: Install Python Dependencies

```bash
cd RAG
pip install -r requirements.txt
```

### Step 3: Add Your PDF

Place any PDF file in the `RAG` folder. Examples:
- A movie script (Inception, Interstellar)
- An anime wiki page exported as PDF
- Your favorite book chapter

### Step 4: Run the Chatbot!

```bash
python fandom_chatbot.py
```

You'll be prompted to enter the PDF filename, then you can ask questions!

---

## 🧠 How RAG Works (The GPS Analogy)

Imagine you're a tourist in a new city:

1. **Document Loading** = Getting the city map (your PDF)
2. **Text Splitting** = Breaking the map into neighborhoods (chunks)
3. **Embeddings** = Giving each neighborhood GPS coordinates (vectors)
4. **Vector Store** = A super-fast GPS that finds relevant neighborhoods
5. **LLM** = A local guide who explains what's in those neighborhoods

When you ask a question:
1. Your question gets converted to GPS coordinates
2. The vector store finds the closest neighborhoods
3. The LLM reads those neighborhoods and answers your question!

---

## 📁 Project Structure

```
RAG/
├── fandom_chatbot.py   # Main RAG pipeline
├── requirements.txt    # Python dependencies
├── README.md          # You are here!
└── your_file.pdf      # Your PDF goes here
```

---

## 🐛 Troubleshooting

### "Ollama not running" error
```bash
# Start the Ollama service
ollama serve
```

### "Model not found" error
```bash
# Pull the model first
ollama pull llama3.2
```

### Slow first run?
The first time you run, it downloads the embedding model (~90MB). This is cached for future runs!

---

## 🎯 Challenge Ideas

1. **Movie Script Bot:** Upload a screenplay and ask about plot points
2. **Anime Wiki Bot:** Export a fandom wiki page and quiz it
3. **Textbook Helper:** Upload lecture notes and get explanations

---

Made with ❤️ for Gen Z CS Students in Kerala
