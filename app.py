import os
import gradio as gr
import re
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import textwrap

# -----------------------------
# Wrap text box
# -----------------------------
def wrap_text(text, width=40):
    return "<br>".join(textwrap.wrap(text, width))

# -----------------------------
# Models
# -----------------------------
MODELS = {"MiniLM": "all-MiniLM-L6-v2", "MPNet": "all-mpnet-base-v2"}

# -----------------------------
# Load data
# -----------------------------
def load_data(folder_path="data"):
    texts = []
    sources = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(content)
                sources.append(filename)
    return texts, sources

def get_sources():
    # change dataset HERE
    _, sources = load_data("./datasets/rijn")
    return sorted(list(set(sources)))  
# -----------------------------
# Simple sentence tokenizer (no nltk)
# -----------------------------
def sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# -----------------------------
# Semantic chunking helper
# -----------------------------
def semantic_chunking(text, model, max_chunk_size=5, similarity_threshold=0.7):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )

        if sim > similarity_threshold and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -----------------------------
# Main chunking function
# -----------------------------
def chunk_text(texts, sources, method, chunk_size, overlap=20, model=None):
    chunks = []
    chunk_sources = []

    for text, source in zip(texts, sources):

        # -----------------------------
        # CHARACTER CHUNKING
        # -----------------------------
        if method == "characters":
            stride = max(chunk_size - overlap, 1)
            for i in range(0, len(text), stride):
                chunk = text[i:i+chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
                    chunk_sources.append(source)

        # -----------------------------
        # WORD CHUNKING
        # -----------------------------
        elif method == "words":
            words = text.split()
            stride = max(chunk_size - overlap, 1)

            for i in range(0, len(words), stride):
                chunk = " ".join(words[i:i+chunk_size])
                if chunk:
                    chunks.append(chunk)
                    chunk_sources.append(source)

        # -----------------------------
        # SENTENCE CHUNKING
        # -----------------------------
        elif method == "sentences":
            sentences = sent_tokenize(text)

            for i in range(0, len(sentences), chunk_size):
                chunk = " ".join(sentences[i:i+chunk_size])
                if chunk:
                    chunks.append(chunk)
                    chunk_sources.append(source)

        # -----------------------------
        # SEMANTIC CHUNKING
        # -----------------------------
        elif method == "semantic":
            semantic_chunks = semantic_chunking(
                text,
                model=model,
                max_chunk_size=chunk_size
            )

            for chunk in semantic_chunks:
                chunks.append(chunk)
                chunk_sources.append(source)

        else:
            raise ValueError(f"Unknown chunking method: {method}")

    return chunks, chunk_sources

# -----------------------------
# Cluster labeling
# -----------------------------
def generate_cluster_labels(df, top_n=3):
    cluster_names = {}

    for cluster_id in df["cluster"].unique():
        texts = df[df["cluster"] == cluster_id]["text"]

        if len(texts) < 2:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
            continue

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50
        )
        X = vectorizer.fit_transform(texts)

        scores = np.asarray(X.mean(axis=0)).flatten()
        terms = np.array(vectorizer.get_feature_names_out())

        top_terms = terms[np.argsort(scores)[-top_n:]]
        cluster_names[cluster_id] = ", ".join(top_terms)

    return cluster_names

# -----------------------------
# Build pipeline
# -----------------------------
def build_system(model_name, chunk_method, chunk_size, overlap=5):
    texts, sources = load_data("./datasets/rijn")
    model = SentenceTransformer(MODELS[model_name])
    chunks, chunk_sources = chunk_text(
        texts,
        sources,
        method=chunk_method,
        chunk_size=chunk_size,
        overlap=overlap,
        model=model if chunk_method == "semantic" else None
    )

    embeddings = model.encode(chunks)
    embeddings = normalize(embeddings)

    # FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # UMAP - 3D
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=10,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    emb_3d = reducer.fit_transform(embeddings)

    # Clustering
    kmeans = KMeans(n_clusters=min(5, len(chunks)), random_state=0)
    labels = kmeans.fit_predict(embeddings)

    df = pd.DataFrame({
        "x": emb_3d[:, 0],
        "y": emb_3d[:, 1],
        "z": emb_3d[:, 2],
        "text": chunks,
        "source": chunk_sources,
        "cluster": labels
    })
    df["wrapped_text"] = df["text"].apply(lambda x: wrap_text(x, 40))

    return model, index, df, reducer

# -----------------------------
# Main function
# -----------------------------
def run(query, model_name, k, chunk_method, chunk_size, overlap, color_mode, selected_sources, show_labels):
    model, index, df, reducer = build_system(
        model_name, chunk_method, chunk_size, overlap
    )

    # -----------------------------
    # FILTER BY SOURCE
    # -----------------------------
    if selected_sources:
        df = df[df["source"].isin(selected_sources)]

    # -----------------------------
    # QUERY
    # -----------------------------
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)
    results = [df.iloc[i]["text"] for i in I[0] if i < len(df)]

    query_3d = reducer.transform(query_embedding)
    df_query = pd.DataFrame({
        "x": [query_3d[0][0]],
        "y": [query_3d[0][1]],
        "z": [query_3d[0][2]],
        "text": [query],
        "source": ["query"],
        "cluster": [-1]
    })

    df["type"] = "document"
    df_query["type"] = "query"

    df_all = pd.concat([df, df_query])

    # -----------------------------
    # CLUSTER LABELS
    # -----------------------------
    if show_labels and color_mode == "cluster":
        cluster_names = generate_cluster_labels(df)
        df_all["cluster_label"] = df_all["cluster"].map(cluster_names)
        color_field = "cluster_label"
    else:
        color_field = color_mode

    # -----------------------------
    # PLOT
    # -----------------------------
    fig = px.scatter_3d(
        df_all,
        x="x",
        y="y",
        z="z",
        color=color_field,
        # symbol="type",
        hover_data=["wrapped_text", "source"],
        title=f"3D Embedding Space (colored by {color_field})"
    )

    fig.update_layout(
        height=800,
        autosize=True,
        margin=dict(l=5, r=5, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
            )
        )

    return fig, "\n".join(results)

# -----------------------------
# inspection
# -----------------------------
def inspect_point(point_id, model_name, chunk_method, chunk_size):
    model, index, df, _ = build_system(model_name, chunk_method, chunk_size)

    point_id = int(point_id)
    selected_text = df.iloc[point_id]["text"]

    query_embedding = model.encode([selected_text])
    D, I = index.search(query_embedding, 3)
    results = [df.iloc[i]["text"] for i in I[0]]

    return f"Selected:\n{selected_text}\n\nSimilar:\n" + "\n".join(results)

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(fill_width=True) as app:

    gr.Markdown("# RAG Embedding Explorer")
    with gr.Row():
        with gr.Column(scale=1):
            model_name = gr.Dropdown(
                choices=list(MODELS.keys()), value="MiniLM", label="Embedding Model"
            )

            k = gr.Slider(1, 5, value=3, step=1, label="Top-K Retrieval")

            chunk_method = gr.Radio(
                choices=["characters", "words", "sentences", "semantic"],
                value="words",
                label="Chunking Method"
                )

            chunk_size = gr.Slider(10, 500, value=50, step=10, label="Chunk Size (chars / words / sentences / semantic max)")
            overlap = gr.Slider(0, 100, value=20, step=5, label="Overlap (chars/words only)")
            color_mode = gr.Radio(
                choices=["cluster", "source"],
                value="cluster",
                label="Color By"
                )
            source_filter = gr.Dropdown(
                choices=get_sources(),   # ✅ directly pass list
                value=get_sources(),     # optional: select all by default
                multiselect=True,
                label="Filter by Source"
                )

            show_labels = gr.Checkbox(
                value=False,
                label="Show Cluster Labels (auto topics)"
                )
            query = gr.Textbox(label="Query", value="outdoor activities")

            run_btn = gr.Button("Run")

        with gr.Column(scale=3):
            plot = gr.Plot(label="Embedding Space")
            output = gr.Textbox(label="Results")
            
            run_btn.click(
                fn=run,
                inputs=[
                    query,
                    model_name,
                    k,
                    chunk_method,
                    chunk_size,
                    overlap,
                    color_mode,
                    source_filter,
                    show_labels
                    ],
                outputs=[plot, output]
                )
            
    # selected_id = gr.Number(label="Select point ID", precision=0)
    # inspect_btn = gr.Button("Inspect Point")
    # inspect_btn.click(
    #     fn=inspect_point, inputs=[selected_id, model_name, chunk_method, chunk_size], outputs=output
    # )

app.launch()