# 🏫 Smart Campus Navigator - Exercise 2

> **Workshop:** Bringing Your Data to Life: Dive into RAG  
> **By:** Rizwan Rahim, CTO of DataVine

Building on Exercise 1, this exercise teaches you how to build a **production-ready RAG system** that handles multiple documents and **persists your vector database** to disk.

---

## 🆕 What's New in Exercise 2?

| Feature | Exercise 1 | Exercise 2 |
|---------|-----------|-----------|
| Documents | Single PDF | Multiple files (folder) |
| Vector Store | FAISS (in-memory) | ChromaDB (persistent) |
| Data Loading | PyPDFLoader | DirectoryLoader |
| Re-processing | Every time | Only once! |

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd Exercise2
pip install -r requirements.txt
```

### Step 2: Add Your Campus Data

Place your `.txt` or `.pdf` files in the `Campus_data/` folder:

```
Exercise2/
├── Campus_data/
│   ├── mess_menu.txt
│   ├── hostel_rules.txt
│   └── events.txt
├── campus_navigator.py
└── requirements.txt
```

### Step 3: Run the Navigator

```bash
python campus_navigator.py
```

**First run:** The script will process all documents and create the vector database.  
**Subsequent runs:** The script loads the existing database instantly! 🚀

---

## 💾 Why Persistent Storage Matters

In Exercise 1, we used FAISS which lives in memory. Every time you restart, you have to re-process all your documents.

In production, you want to:
1. Process documents **once**
2. Save the vector database to disk
3. Load it instantly on future runs

ChromaDB does this automatically with `persist_directory`!

---

## 🧠 The Production RAG Pattern

```
┌─────────────────┐
│  Campus Data    │  ◄── Multiple .txt/.pdf files
│  (Folder)       │
└────────┬────────┘
         │
         ▼ DirectoryLoader
┌─────────────────┐
│  Documents      │
└────────┬────────┘
         │
         ▼ Text Splitter
┌─────────────────┐
│  Chunks         │
└────────┬────────┘
         │
         ▼ Embeddings + ChromaDB
┌─────────────────┐
│  Vector Store   │ ◄── Saved to ./chroma_campus_db/
│  (Persistent)   │
└────────┬────────┘
         │
         ▼ Retriever + LLM
┌─────────────────┐
│  RAG Chain      │
└─────────────────┘
```

---

## 💡 Sample Questions to Ask

Try these questions with the campus data:

- "What's for lunch on Thursday?"
- "When does the library close on Saturday?"
- "Tell me about the Tech Fest"
- "What's the curfew time?"
- "When is the DataVine workshop?"

---

## 🔧 Troubleshooting

### "Collection not found" error
Delete the `chroma_campus_db` folder and run again to rebuild.

### Adding new documents
Delete the `chroma_campus_db` folder, add your new files to `Campus_data/`, and run again.

---
