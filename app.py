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
import tempfile

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
def load_data(folder_path="./datasets/rijn"):
    if not os.path.exists(folder_path):
        return ["No data found"], ["none"]
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
    _, sources = load_data()
    return sorted(list(set(sources)))  

# -----------------------------
# Simple sentence tokenizer
# -----------------------------
def sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# -----------------------------
# Semantic chunking helper
# -----------------------------
def semantic_chunking(text, model, max_chunk_size=5, similarity_threshold=0.7):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1: return sentences
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
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

# -----------------------------
# Main chunking function
# -----------------------------
def chunk_text(texts, sources, method, chunk_size, overlap=20, model=None):
    chunks, chunk_sources = [], []
    for text, source in zip(texts, sources):
        if method == "characters":
            stride = max(chunk_size - overlap, 1)
            for i in range(0, len(text), stride):
                chunk = text[i:i+chunk_size]
                if chunk.strip():
                    chunks.append(chunk); chunk_sources.append(source)
        elif method == "words":
            words = text.split()
            stride = max(chunk_size - overlap, 1)
            for i in range(0, len(words), stride):
                chunk = " ".join(words[i:i+chunk_size])
                if chunk: chunks.append(chunk); chunk_sources.append(source)
        elif method == "sentences":
            sentences = sent_tokenize(text)
            for i in range(0, len(sentences), chunk_size):
                chunk = " ".join(sentences[i:i+chunk_size])
                if chunk: chunks.append(chunk); chunk_sources.append(source)
        elif method == "semantic":
            semantic_chunks = semantic_chunking(text, model=model, max_chunk_size=chunk_size)
            for chunk in semantic_chunks:
                chunks.append(chunk); chunk_sources.append(source)
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
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        X = vectorizer.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).flatten()
        terms = np.array(vectorizer.get_feature_names_out())
        top_terms = terms[np.argsort(scores)[-top_n:]]
        cluster_names[cluster_id] = ", ".join(top_terms)
    return cluster_names

# -----------------------------
# Build pipeline
# -----------------------------
def build_system(model_name, chunk_method, chunk_size, overlap=5, num_clusters=5):
    texts, sources = load_data()
    model = SentenceTransformer(MODELS[model_name])
    chunks, chunk_sources = chunk_text(texts, sources, method=chunk_method, chunk_size=chunk_size, overlap=overlap, model=model if chunk_method == "semantic" else None)
    embeddings = normalize(model.encode(chunks))
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    reducer = umap.UMAP(n_components=3, n_neighbors=10, min_dist=0.05, metric="cosine", random_state=42)
    emb_3d = reducer.fit_transform(embeddings)
    actual_clusters = min(num_clusters, len(chunks))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    df = pd.DataFrame({"x": emb_3d[:, 0], "y": emb_3d[:, 1], "z": emb_3d[:, 2], "text": chunks, "source": chunk_sources, "cluster": labels})
    df["wrapped_text"] = df["text"].apply(lambda x: wrap_text(x, 40))
    return model, index, df, reducer

# -----------------------------
# Main function
# -----------------------------
def run(query, model_name, k, chunk_method, chunk_size, overlap, color_mode, selected_sources, show_labels, num_clusters):
    model, index, df, reducer = build_system(model_name, chunk_method, chunk_size, overlap, num_clusters)
    if selected_sources:
        df = df[df["source"].isin(selected_sources)]
    
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)
    results = [df.iloc[i]["text"] for i in I[0] if i < len(df)]
    query_3d = reducer.transform(query_embedding)
    
    df_query = pd.DataFrame({"x": [query_3d[0][0]], "y": [query_3d[0][1]], "z": [query_3d[0][2]], "text": [query], "source": ["query"], "cluster": [-1], "type": ["query"]})
    df["type"] = "document"
    df_all = pd.concat([df, df_query])

    if show_labels and color_mode == "cluster":
        cluster_names = generate_cluster_labels(df)
        df_all["cluster_label"] = df_all["cluster"].map(cluster_names).fillna("Query")
        color_field = "cluster_label"
    else:
        color_field = color_mode
    
    fig = px.scatter_3d(df_all, x="x", y="y", z="z", color=color_field, hover_data=["wrapped_text", "source"], title="3D Embedding Space")
    fig.update_layout(height=800, margin=dict(l=0, r=0, t=30, b=0), scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))))

    html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')

    controls_and_script = """
    <div id="rotation-controls" style="position: absolute; top: 40px; left: 10px; z-index: 100; background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; font-family: sans-serif; box-shadow: 0 4px 15px rgba(0,0,0,0.2); width: 220px;">
        <h4 style="margin: 0 0 10px 0; color: #333;">3D Navigation</h4>
        <button id="toggle-btn" style="width: 100%; padding: 10px; margin-bottom: 10px; cursor: pointer; border: none; border-radius: 4px; background: #f44336; color: white; font-weight: bold;">Stop Rotation</button>
        <div id="status-indicator" style="font-size: 11px; font-weight: bold; text-align: center; margin-bottom: 15px; color: #d32f2f;">ZOOM LOCKED</div>
        <label style="display: block; font-size: 11px; color: #666; margin-bottom: 5px;">SPEED</label>
        <input type="range" id="speed-slider" min="0.001" max="0.05" step="0.001" value="0.01" style="width: 100%; margin-bottom: 15px;">
        <label style="display: block; font-size: 11px; color: #666; margin-bottom: 5px;">AXIS</label>
        <select id="axis-select" style="width: 100%; padding: 8px; cursor: pointer; border: 1px solid #ddd; border-radius: 4px; background: white;">
            <option value="z">Z-Axis (Turntable)</option>
            <option value="y">Y-Axis (Vertical)</option>
            <option value="x">X-Axis (Side-to-Side)</option>
        </select>
    </div>
    <script>
    (function() {
        var angle = 0, isRotating = true, speed = 0.01, axis = 'z', radius = 1.8;
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        var toggleBtn = document.getElementById('toggle-btn'), speedSlider = document.getElementById('speed-slider');
        var axisSelect = document.getElementById('axis-select'), statusInd = document.getElementById('status-indicator');

        function setZoomEnabled(enabled) {
            Plotly.relayout(plotDiv, {'scene.xaxis.fixedrange': !enabled, 'scene.yaxis.fixedrange': !enabled, 'scene.zaxis.fixedrange': !enabled});
            statusInd.innerHTML = enabled ? "ZOOM ENABLED" : "ZOOM LOCKED";
            statusInd.style.color = enabled ? "#2e7d32" : "#d32f2f";
        }

        function rotate() {
            if (!isRotating) return;
            angle += parseFloat(speed);
            var curr = plotDiv._fullLayout.scene.camera.eye;
            var eye = {};
            if (axis === 'z') eye = { x: radius * Math.cos(angle), y: radius * Math.sin(angle), z: curr.z };
            else if (axis === 'y') eye = { x: radius * Math.cos(angle), y: curr.y, z: radius * Math.sin(angle) };
            else eye = { x: curr.x, y: radius * Math.cos(angle), z: radius * Math.sin(angle) };
            Plotly.relayout(plotDiv, { 'scene.camera.eye': eye });
            requestAnimationFrame(rotate);
        }

        toggleBtn.onclick = function() {
            isRotating = !isRotating;
            if (isRotating) {
                var e = plotDiv._fullLayout.scene.camera.eye;
                radius = Math.sqrt(e.x*e.x + e.y*e.y + e.z*e.z);
                angle = Math.atan2(e.y, e.x);
                this.innerHTML = "Stop Rotation"; this.style.background = "#f44336";
                setZoomEnabled(false); rotate();
            } else {
                this.innerHTML = "Start Rotation"; this.style.background = "#4CAF50";
                setZoomEnabled(true);
            }
        };
        speedSlider.oninput = function() { speed = this.value; };
        axisSelect.onchange = function() { axis = this.value; };
        setTimeout(function() { 
            var e = plotDiv._fullLayout.scene.camera.eye;
            radius = Math.sqrt(e.x*e.x + e.y*e.y + e.z*e.z);
            setZoomEnabled(false); rotate(); 
        }, 2000);
    })();
    </script>
    """
    full_html = html_str.replace("</body>", controls_and_script + "</body>")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    with open(tmp.name, "w", encoding="utf-8") as f: f.write(full_html)
    return fig, "\n".join(results), tmp.name

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(fill_width=True) as app:
    gr.Markdown("# RAG Embedding Explorer")
    with gr.Row():
        with gr.Column(scale=1):
            model_name = gr.Dropdown(choices=list(MODELS.keys()), value="MiniLM", label="Embedding Model")
            num_clusters = gr.Slider(2, 20, value=5, step=1, label="Number of Clusters")
            k = gr.Slider(1, 5, value=3, step=1, label="Top-K Retrieval")
            chunk_method = gr.Radio(choices=["characters", "words", "sentences", "semantic"], value="words", label="Chunking Method")
            chunk_size = gr.Slider(10, 500, value=50, step=10, label="Size")
            overlap = gr.Slider(0, 100, value=20, step=5, label="Overlap")
            color_mode = gr.Radio(choices=["cluster", "source"], value="cluster", label="Color By")
            source_filter = gr.Dropdown(choices=get_sources(), value=get_sources(), multiselect=True, label="Sources")
            show_labels = gr.Checkbox(value=False, label="Show Labels")
            query = gr.Textbox(label="Query", value="outdoor activities")
            run_btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=3):
            plot = gr.Plot(label="Embedding Space")
            full_screen_link = gr.File(label="Download Interactive HTML", visible=False)
            output = gr.Textbox(label="Results", lines=10)

    def on_run_click(*args):
        fig, results, html_path = run(*args)
        return fig, results, gr.update(value=html_path, visible=True)

    run_btn.click(
        fn=on_run_click,
        inputs=[query, model_name, k, chunk_method, chunk_size, overlap, color_mode, source_filter, show_labels, num_clusters],
        outputs=[plot, output, full_screen_link]
    )

app.launch()