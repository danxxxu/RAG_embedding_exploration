from sentence_transformers import SentenceTransformer

MODELS = {
    "MiniLM": "all-MiniLM-L6-v2",
    "MPNet": "all-mpnet-base-v2"
}

def load_model(name="MiniLM"):
    return SentenceTransformer(MODELS[name])

def embed_texts(model, texts):
    return model.encode(texts)