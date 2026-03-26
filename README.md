# RAG Embedding Explorer

This is a **local RAG (Retrieval-Augmented Generation) embedding explorer** for text datasets.  
It allows you to:

- Load multiple text files from a folder
- Chunk text in multiple ways (characters, words, sentences, semantic)
- Embed chunks using `SentenceTransformers` models
- Reduce embeddings to 3D using `UMAP`
- Cluster embeddings with `KMeans` and generate auto topic labels
- Visualize interactive 3D embeddings with Plotly
- Filter by source files or cluster
- Perform query retrieval with top-k nearest chunks (FAISS)

---

## Features

- Multiple chunking options: characters, words, sentences, semantic
- Overlap support for characters and words
- Interactive 3D Plotly visualization
- Color by cluster or source
- Auto-generated cluster labels using top terms
- Filterable by source file
- Live query input to see nearest chunks

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_folder>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt