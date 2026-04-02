import os
import gradio as gr
import re
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import textwrap
import tempfile
import joblib
from PIL import Image
from pillow_heif import register_heif_opener
import base64
from io import BytesIO
import plotly.graph_objects as go
import copy

register_heif_opener()

BASE_DATASETS_DIR = "./datasets"


def list_available_datasets():
    if not os.path.exists(BASE_DATASETS_DIR):
        os.makedirs(BASE_DATASETS_DIR)
    folders = [
        f
        for f in os.listdir(BASE_DATASETS_DIR)
        if os.path.isdir(os.path.join(BASE_DATASETS_DIR, f))
    ]
    return sorted(folders)


# -----------------------------
# Configuration & Models
# -----------------------------
MODELS = {
    "MiniLM (Text Only)": "all-MiniLM-L6-v2",
    "MPNet (Text Only)": "all-mpnet-base-v2",
    "CLIP (Multimodal)": "clip-ViT-B-32",
}

SOURCE_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
]


# -----------------------------
# Session persistence
# -----------------------------
def save_session(df, index, reducer, dataset_name, folder="session_data"):
    path = os.path.join(folder, dataset_name)
    os.makedirs(path, exist_ok=True)
    df.drop(columns=["thumbnail"], errors="ignore").to_parquet(
        os.path.join(path, "data.parquet")
    )
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    joblib.dump(reducer, os.path.join(path, "reducer.joblib"))
    print("Session saved to disk.")


def load_session(dataset_name, folder="session_data"):
    path = os.path.join(folder, dataset_name)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(os.path.join(path, "data.parquet"))
            index = faiss.read_index(os.path.join(path, "index.faiss"))
            reducer = joblib.load(os.path.join(path, "reducer.joblib"))
            return df, index, reducer
        except Exception as e:
            print(f"Failed to load session: {e}")
    return None, None, None


# -----------------------------
# Text helpers
# -----------------------------
def wrap_text(text, width=40):
    return "<br>".join(textwrap.wrap(text, width))


def sent_tokenize(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# -----------------------------
# Multimodal Data Loading
# -----------------------------
def load_multimodal_data(folder_path, is_clip=False):
    if not os.path.exists(folder_path):
        return [], [], []

    items, sources, item_types = [], [], []
    image_exts = (".jpg", ".jpeg", ".png", ".heic")

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                items.append(f.read())
                sources.append(filename)
                item_types.append("text")
        elif is_clip and filename.lower().endswith(image_exts):
            try:
                img = Image.open(filepath).convert("RGB")
                items.append(img)
                sources.append(filename)
                item_types.append("image")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return items, sources, item_types


# -----------------------------
# Chunking
# -----------------------------
def semantic_chunking(text, model, max_chunk_size=5, similarity_threshold=0.7):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences
    embeddings = model.encode(sentences)
    chunks, current_chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
        )
        if sim > similarity_threshold and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def chunk_text(texts, sources, method, chunk_size, overlap=20, model=None):
    chunks, chunk_sources = [], []
    for text, source in zip(texts, sources):
        if method == "characters":
            stride = max(chunk_size - overlap, 1)
            for i in range(0, len(text), stride):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
                    chunk_sources.append(source)
        elif method == "words":
            words = text.split()
            stride = max(chunk_size - overlap, 1)
            for i in range(0, len(words), stride):
                chunk = " ".join(words[i : i + chunk_size])
                if chunk:
                    chunks.append(chunk)
                    chunk_sources.append(source)
        elif method == "sentences":
            sentences = sent_tokenize(text)
            for i in range(0, len(sentences), chunk_size):
                chunk = " ".join(sentences[i : i + chunk_size])
                if chunk:
                    chunks.append(chunk)
                    chunk_sources.append(source)
        elif method == "semantic":
            for chunk in semantic_chunking(
                text, model=model, max_chunk_size=chunk_size
            ):
                chunks.append(chunk)
                chunk_sources.append(source)
    return chunks, chunk_sources


# -----------------------------
# Cluster labeling (TF-IDF on text only)
# -----------------------------
def generate_cluster_labels(df, top_n=3):
    cluster_names = {}
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[(df["cluster"] == cluster_id) & (df["data_type"] == "text")]
        if len(subset) < 2:
            cluster_names[cluster_id] = f"Cluster {cluster_id}: (no text)"
            continue
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        X = vectorizer.fit_transform(subset["text"])
        scores = np.asarray(X.mean(axis=0)).flatten()
        terms = np.array(vectorizer.get_feature_names_out())
        top_terms = terms[np.argsort(scores)[-top_n:]]
        cluster_names[cluster_id] = f"Cluster {cluster_id}: " + ", ".join(top_terms)
    return cluster_names


# -----------------------------
# Map cluster id → Viridis hex colour
# Matches the Viridis colorscale used for scatter markers exactly.
# -----------------------------
def viridis_hex(cluster_id, cluster_min, cluster_max):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    span = max(cluster_max - cluster_min, 1)
    r, g, b, _ = cmap((cluster_id - cluster_min) / span)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# -----------------------------
# Image thumbnail → base64
# -----------------------------
def get_thumbnail_base64(path, size=(80, 80)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return "data:image/jpeg;base64," + "".join(img_str.split())
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""


# -----------------------------
# export json
# -----------------------------
def export_to_json(df, filename="embedding_space.json"):
    import json
    import os

    # ✅ Only keep what you NEED
    df_export = df[["x", "y", "z", "cluster", "source", "data_type"]].copy()

    # Optional: include short text preview (SAFE)
    df_export["text"] = df["text"].astype(str).str.slice(0, 200)

    # Normalize safely
    for axis in ["x", "y", "z"]:
        std = df_export[axis].std()
        if std != 0:
            df_export[axis] = (df_export[axis] - df_export[axis].mean()) / std

    export_data = df_export.to_dict(orient="records")

    save_path = os.path.join(".", filename)

    print(f"Exporting {len(export_data)} points...")  # ✅ debug

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f)

    print("Export done.")  # ✅ debug

    return save_path


def on_export_click(filename):
    global CURRENT_DF

    if CURRENT_DF is None or len(CURRENT_DF) == 0:
        return None

    # Ensure it ends with .json
    if not filename.endswith(".json"):
        filename += ".json"

    path = export_to_json(CURRENT_DF, filename)

    return path


# -----------------------------
# Build pipeline
# -----------------------------
def build_system(
    dataset_name, model_name, chunk_method, chunk_size, overlap=5, num_clusters=5
):
    current_path = os.path.join(BASE_DATASETS_DIR, dataset_name)
    is_clip = "CLIP" in model_name
    raw_items, raw_sources, raw_types = load_multimodal_data(
        folder_path=current_path, is_clip=is_clip
    )

    model = SentenceTransformer(MODELS[model_name])
    final_data, final_sources, final_types, display_texts, thumbnails = (
        [],
        [],
        [],
        [],
        [],
    )

    for item, source, itype in zip(raw_items, raw_sources, raw_types):
        if itype == "text":
            chunks = chunk_text(
                [item],
                [source],
                method=chunk_method,
                chunk_size=chunk_size,
                overlap=overlap,
                model=model if chunk_method == "semantic" else None,
            )[0]
            for c in chunks:
                final_data.append(c)
                final_sources.append(source)
                final_types.append("text")
                display_texts.append(c)
                thumbnails.append("")
        else:
            final_data.append(item)
            final_sources.append(source)
            final_types.append("image")
            display_texts.append(f"[IMAGE: {source}]")
            thumbnails.append(get_thumbnail_base64(os.path.join(current_path, source)))

    embeddings = normalize(model.encode(final_data, show_progress_bar=True))

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    reducer = umap.UMAP(
        n_components=3, n_neighbors=10, min_dist=0.05, metric="cosine", random_state=42
    )
    emb_3d = reducer.fit_transform(embeddings)

    actual_clusters = min(num_clusters, len(final_data))
    labels = KMeans(n_clusters=actual_clusters, n_init=10, random_state=0).fit_predict(
        emb_3d
    )

    df = pd.DataFrame(
        {
            "x": emb_3d[:, 0],
            "y": emb_3d[:, 1],
            "z": emb_3d[:, 2],
            "text": display_texts,
            "source": final_sources,
            "cluster": labels,
            "data_type": final_types,
            "thumbnail": thumbnails,
        }
    )
    df["wrapped_text"] = df["text"].apply(lambda x: wrap_text(str(x), 40))
    return model, index, df, reducer


# -----------------------------
# Build Plotly figure
# -----------------------------
def build_figure(df, query_3d, color_mode, show_labels, for_gradio=False):
    """
    color_mode  : "cluster" | "source"
    show_labels : populate legend entries with TF-IDF cluster keywords
    for_gradio  : True  → Plotly native hover for text; thumbnails stripped
                  False → ALL Plotly native hovers suppressed; JS tooltip handles everything
    """
    fig = go.Figure()

    cluster_min = int(df["cluster"].min())
    cluster_max = int(df["cluster"].max())

    unique_sources = sorted(df["source"].unique())
    source_color_map = {
        s: SOURCE_COLORS[i % len(SOURCE_COLORS)] for i, s in enumerate(unique_sources)
    }

    df_text = df[df["data_type"] == "text"]
    df_img = df[df["data_type"] == "image"]

    # ── Helper: marker dict ───────────────────────────────────
    def make_marker(series, size, symbol=None):
        base = dict(size=size, opacity=0.9)
        if symbol:
            base["symbol"] = symbol
            base["line"] = dict(width=1, color="white")
        if color_mode == "cluster":
            base.update(
                dict(
                    color=series["cluster"],
                    colorscale="Viridis",
                    cmin=cluster_min,
                    cmax=cluster_max,
                    showscale=False,
                )
            )
        else:
            base["color"] = series["source"].map(source_color_map).tolist()
        return base

    # ── Legend-only traces (one per cluster or source) ────────
    # These are invisible scatter points whose only job is to appear
    # in the legend with the right colour and label.
    if color_mode == "cluster":
        label_map = (
            generate_cluster_labels(df)
            if show_labels
            else {cid: f"Cluster {cid}" for cid in sorted(df["cluster"].unique())}
        )
        for cid in sorted(df["cluster"].unique()):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(
                        size=8, color=viridis_hex(cid, cluster_min, cluster_max)
                    ),
                    name=label_map.get(cid, f"Cluster {cid}"),
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
    else:
        for src, color in source_color_map.items():
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=8, color=color),
                    name=src,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    # ── Text points ───────────────────────────────────────────
    # customdata: [source, wrapped_text]
    # Gradio → native Plotly hover (shows wrapped_text correctly)
    # HTML   → suppress native hover; JS tooltip reads the same customdata
    if not df_text.empty:
        fig.add_trace(
            go.Scatter3d(
                x=df_text["x"],
                y=df_text["y"],
                z=df_text["z"],
                mode="markers",
                marker=make_marker(df_text, size=3),
                name="Text Docs",
                showlegend=False,
                customdata=np.stack(
                    (df_text["source"], df_text["wrapped_text"]), axis=-1
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
                    if for_gradio
                    else "<extra></extra>"  # hides native tooltip; JS uses customdata directly
                ),
            )
        )

    # ── Image points ──────────────────────────────────────────
    # customdata: [source, thumbnail_b64]
    # Gradio → strip thumbnail to avoid JSON hang; show filename via native hover
    # HTML   → keep thumbnail; suppress native hover; JS tooltip shows image
    if not df_img.empty:
        if for_gradio:
            img_customdata = np.stack(
                (df_img["source"], np.full(len(df_img), "")), axis=-1
            )
            img_hovertemplate = "<b>%{customdata[0]}</b><extra></extra>"
        else:
            img_customdata = np.stack((df_img["source"], df_img["thumbnail"]), axis=-1)
            img_hovertemplate = "<extra></extra>"

        fig.add_trace(
            go.Scatter3d(
                x=df_img["x"],
                y=df_img["y"],
                z=df_img["z"],
                mode="markers",
                name="Images",
                showlegend=False,
                marker=make_marker(df_img, size=7, symbol="diamond"),
                customdata=img_customdata,
                hovertemplate=img_hovertemplate,
            )
        )

    # ── Query point ───────────────────────────────────────────
    fig.add_trace(
        go.Scatter3d(
            x=[query_3d[0][0]],
            y=[query_3d[0][1]],
            z=[query_3d[0][2]],
            mode="markers+text",
            marker=dict(size=10, color="white", line=dict(width=2, color="black")),
            text=["SEARCH QUERY"],
            textfont=dict(color="white", size=11),
            name="Query",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            bgcolor="rgba(15,15,25,1)",
            xaxis=dict(gridcolor="#333", color="#aaa"),
            yaxis=dict(gridcolor="#333", color="#aaa"),
            zaxis=dict(gridcolor="#333", color="#aaa"),
        ),
        legend=dict(
            bgcolor="rgba(30,30,40,0.85)",
            font=dict(color="white", size=11),
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            itemsizing="constant",
        ),
    )
    return fig


# -----------------------------
# HTML overlay: unified JS tooltip + rotation controls
#
# The JS tooltip handles BOTH text and image points in the HTML output:
#   customdata[0] → source filename  (always present)
#   customdata[1] → wrapped_text for text points, base64 URI for images
# Detection is done by checking whether the string starts with "data:image".
# -----------------------------
CONTROLS_AND_SCRIPT = """
<div id="js-tooltip" style="
    display:none; position:fixed; z-index:9999;
    background:rgba(20,20,30,0.95); color:#eee;
    border:1px solid #555; border-radius:8px; padding:10px;
    box-shadow:0 4px 20px rgba(0,0,0,0.5);
    pointer-events:none; max-width:260px;
    font-family:sans-serif; font-size:12px; line-height:1.5;">
</div>

<div id="rotation-controls" style="position:fixed; top:60px; left:10px; z-index:100;
    background:rgba(255,255,255,0.92); padding:15px; border-radius:8px;
    font-family:sans-serif; box-shadow:0 4px 15px rgba(0,0,0,0.2); width:220px;">
    <h4 style="margin:0 0 10px 0; color:#333;">3D Navigation</h4>
    <button id="toggle-btn" style="width:100%; padding:10px; margin-bottom:10px;
        cursor:pointer; border:none; border-radius:4px;
        background:#f44336; color:white; font-weight:bold;">Stop Rotation</button>
    <div id="status-indicator" style="font-size:11px; font-weight:bold;
        text-align:center; margin-bottom:15px; color:#d32f2f;">ZOOM LOCKED</div>
    <label style="display:block; font-size:11px; color:#666; margin-bottom:5px;">SPEED</label>
    <input type="range" id="speed-slider" min="0.001" max="0.05"
        step="0.001" value="0.01" style="width:100%; margin-bottom:15px;">
    <label style="display:block; font-size:11px; color:#666; margin-bottom:5px;">AXIS</label>
    <select id="axis-select" style="width:100%; padding:8px; cursor:pointer;
        border:1px solid #ddd; border-radius:4px; background:white;">
        <option value="z">Z-Axis (Turntable)</option>
        <option value="y">Y-Axis (Vertical)</option>
        <option value="x">X-Axis (Side-to-Side)</option>
    </select>
</div>

<script>
(function () {
    var angle = 0, isRotating = true, speed = 0.01, axis = 'z', radius = 1.8;
    var tooltip     = document.getElementById('js-tooltip');
    var toggleBtn   = document.getElementById('toggle-btn');
    var speedSlider = document.getElementById('speed-slider');
    var axisSelect  = document.getElementById('axis-select');
    var statusInd   = document.getElementById('status-indicator');

    function getPlot() { return document.getElementsByClassName('plotly-graph-div')[0]; }

    function setZoomEnabled(enabled) {
        var p = getPlot(); if (!p) return;
        Plotly.relayout(p, {
            'scene.xaxis.fixedrange': !enabled,
            'scene.yaxis.fixedrange': !enabled,
            'scene.zaxis.fixedrange': !enabled
        });
        statusInd.innerHTML   = enabled ? 'ZOOM ENABLED'  : 'ZOOM LOCKED';
        statusInd.style.color = enabled ? '#2e7d32' : '#d32f2f';
    }

    function rotate() {
        if (!isRotating) return;
        var p = getPlot();
        if (!p || !p._fullLayout) { requestAnimationFrame(rotate); return; }
        angle += parseFloat(speed);
        var curr = p._fullLayout.scene.camera.eye;
        var eye  = {};
        if      (axis === 'z') eye = { x: radius*Math.cos(angle), y: radius*Math.sin(angle), z: curr.z };
        else if (axis === 'y') eye = { x: radius*Math.cos(angle), y: curr.y,                 z: radius*Math.sin(angle) };
        else                   eye = { x: curr.x,                 y: radius*Math.cos(angle), z: radius*Math.sin(angle) };
        Plotly.relayout(p, { 'scene.camera.eye': eye });
        requestAnimationFrame(rotate);
    }

    toggleBtn.onclick = function () {
        isRotating = !isRotating;
        var p = getPlot();
        if (isRotating) {
            var e = p._fullLayout.scene.camera.eye;
            radius = Math.sqrt(e.x*e.x + e.y*e.y + e.z*e.z);
            angle  = Math.atan2(e.y, e.x);
            this.innerHTML = 'Stop Rotation';  this.style.background = '#f44336';
            setZoomEnabled(false); rotate();
        } else {
            this.innerHTML = 'Start Rotation'; this.style.background = '#4CAF50';
            setZoomEnabled(true);
        }
    };
    speedSlider.oninput = function () { speed = this.value; };
    axisSelect.onchange  = function () { axis  = this.value; };

    var checkReady = setInterval(function () {
        var p = getPlot();
        if (!p || !p._fullData) return;
        clearInterval(checkReady);

        var e = p._fullLayout.scene.camera.eye;
        radius = Math.sqrt(e.x*e.x + e.y*e.y + e.z*e.z);
        setZoomEnabled(false);
        rotate();

        p.on('plotly_hover', function (eventData) {
            var pt = eventData.points[0];
            if (!pt.customdata || pt.customdata.length < 2) return;

            var label = pt.customdata[0];  // source filename
            var field = pt.customdata[1];  // wrapped_text OR base64 URI

            var html = '<b style="font-size:13px;">' + label + '</b>';

            if (field && field.indexOf('data:image') === 0) {
                // Image point → thumbnail
                html += '<br><img src="' + field +
                        '" style="width:200px;height:auto;margin-top:6px;border-radius:4px;">';
            } else if (field && field.length > 0) {
                // Text point → chunk text
                html += '<br><span style="color:#ccc;">' + field + '</span>';
            }

            tooltip.innerHTML     = html;
            tooltip.style.display = 'block';
        });

        p.on('plotly_unhover', function () { tooltip.style.display = 'none'; });

        document.addEventListener('mousemove', function (ev) {
            if (tooltip.style.display === 'none') return;
            var x = ev.clientX + 15, y = ev.clientY + 15;
            if (x + 280 > window.innerWidth)  x = ev.clientX - 280;
            if (y + 280 > window.innerHeight) y = ev.clientY - 280;
            tooltip.style.left = x + 'px';
            tooltip.style.top  = y + 'px';
        });
    }, 300);
}());
</script>
"""
CURRENT_DF = None


# -----------------------------
# Main run function
# -----------------------------
def run(
    dataset_name,
    query,
    model_name,
    k,
    chunk_method,
    chunk_size,
    overlap,
    color_mode,
    selected_sources,
    show_labels,
    num_clusters,
    force_rebuild,
):
    session_folder = "session_data"
    df, index, reducer = load_session(dataset_name, session_folder)
    model = SentenceTransformer(MODELS[model_name])
    current_dim = model.get_sentence_embedding_dimension()

    if force_rebuild or (df is None) or (index.d != current_dim):
        model, index, df, reducer = build_system(
            dataset_name, model_name, chunk_method, chunk_size, overlap, num_clusters
        )
        save_session(df, index, reducer, dataset_name, session_folder)
    else:
        current_path = os.path.join(BASE_DATASETS_DIR, dataset_name)
        df["thumbnail"] = df.apply(
            lambda row: (
                get_thumbnail_base64(os.path.join(current_path, row["source"]))
                if row["data_type"] == "image"
                else ""
            ),
            axis=1,
        )
        if "wrapped_text" not in df.columns:
            df["wrapped_text"] = df["text"].apply(lambda x: wrap_text(str(x), 40))
    global CURRENT_DF
    CURRENT_DF = df
    df_view = (
        df[df["source"].isin(selected_sources)].copy()
        if selected_sources
        else df.copy()
    )

    query_emb = normalize(model.encode([query]))
    D, I = index.search(query_emb, k)
    results = [df.iloc[i]["text"] for i in I[0] if i < len(df)]
    query_3d = reducer.transform(query_emb)

    # Gradio figure: native Plotly hover for text; thumbnails stripped
    fig_gradio = build_figure(
        df_view, query_3d, color_mode, show_labels, for_gradio=True
    )

    # HTML figure: JS tooltip for all points; full thumbnails kept
    # fig_html  = build_figure(df_view, query_3d, color_mode, show_labels, for_gradio=False)
    # html_str  = fig_html.to_html(full_html=True, include_plotlyjs="cdn")
    fig_html = copy.deepcopy(fig_gradio)
    df_img = df_view[df_view["data_type"] == "image"]
    if not df_img.empty:
        for trace in fig_html.data:
            # Image trace is identified by its name
            if trace.name == "Images" and trace.customdata is not None:
                cd = np.array(trace.customdata, dtype=object)
                cd[:, 1] = df_img["thumbnail"].values  # restore thumbnails
                trace.customdata = cd
                trace.hovertemplate = "<extra></extra>"  # suppress native tooltip
                break
    html_str = fig_html.to_html(full_html=True, include_plotlyjs="cdn")
    full_html = html_str.replace("</body>", CONTROLS_AND_SCRIPT + "</body>")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    with open(tmp.name, "w", encoding="utf-8") as f:
        f.write(full_html)
    return (
        fig_gradio,
        "\n\n---\n\n".join(results),
        gr.update(value=tmp.name, visible=True),
        None,
    )


# -----------------------------
# UI
# -----------------------------
def get_sources_for_dataset(dataset_name):
    if not dataset_name:
        return []
    path = os.path.join(BASE_DATASETS_DIR, dataset_name)
    try:
        return sorted(os.listdir(path))
    except Exception:
        return []


with gr.Blocks(fill_width=True) as app:
    gr.Markdown("# RAG Embedding Explorer")

    with gr.Row():
        # ── Left panel ────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Row():
                available_datasets = list_available_datasets()
                dataset_selector = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Dataset Folder",
                    scale=4,
                )
                refresh_btn = gr.Button("🔄", scale=1, min_width=40)

            model_name = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="MiniLM (Text Only)",
                label="Embedding Model",
            )
            num_clusters = gr.Slider(2, 20, value=5, step=1, label="Number of Clusters")
            k = gr.Slider(1, 10, value=3, step=1, label="Top-K Retrieval")
            chunk_method = gr.Radio(
                choices=["characters", "words", "sentences", "semantic"],
                value="words",
                label="Chunking Method",
            )
            chunk_size = gr.Slider(1, 500, value=50, step=10, label="Chunk Size")
            overlap = gr.Slider(0, 100, value=20, step=5, label="Overlap")

            with gr.Row():
                color_mode = gr.Radio(
                    choices=["cluster", "source"],
                    value="cluster",
                    label="Color By",
                )
                show_labels = gr.Checkbox(value=True, label="Show Cluster Labels")

            _initial_sources = get_sources_for_dataset(
                available_datasets[0] if available_datasets else None
            )
            sources = gr.Dropdown(
                choices=_initial_sources,
                value=_initial_sources,
                multiselect=True,
                label="Filter Sources",
            )
            force_rebuild = gr.Checkbox(
                value=False, label="Force Rebuild Embedding Space"
            )
            query = gr.Textbox(label="Query", value="outdoor activities")
            run_btn = gr.Button("Run", variant="primary")
            export_btn = gr.Button("Export JSON")
            json_file = gr.File(label="Download JSON", visible=False)
            filename_input = gr.Textbox(
                label="Export Filename", value="embedding_space.json"
            )

        # ── Right panel ───────────────────────────────────────
        with gr.Column(scale=3):
            plot = gr.Plot(label="Embedding Space")
            html_file = gr.File(label="Download Interactive HTML", visible=False)
            output = gr.Textbox(label="Top-K Results", lines=10)

    # ── Event handlers ─────────────────────────────────────
    def refresh_folders():
        return gr.update(choices=list_available_datasets())

    def update_sources(dataset_name):
        new_sources = get_sources_for_dataset(dataset_name)
        return gr.update(choices=new_sources, value=new_sources)

    def on_run_click(*args):
        fig, results, html_update, json_path = run(*args)
        return fig, results, html_update, json_path

    refresh_btn.click(fn=refresh_folders, outputs=dataset_selector)
    dataset_selector.change(fn=update_sources, inputs=dataset_selector, outputs=sources)
    run_btn.click(
        fn=on_run_click,
        inputs=[
            dataset_selector,
            query,
            model_name,
            k,
            chunk_method,
            chunk_size,
            overlap,
            color_mode,
            sources,
            show_labels,
            num_clusters,
            force_rebuild,
        ],
        outputs=[plot, output, html_file, json_file],
    )
    export_btn.click(fn=on_export_click, inputs=filename_input, outputs=json_file)

app.launch(theme=gr.themes.Soft())
