import umap
import pandas as pd

def reduce_embeddings(embeddings):
    reducer = umap.UMAP(random_state=42)
    return reducer.fit_transform(embeddings), reducer


def create_dataframe(emb_2d, texts, labels=None):
    df = pd.DataFrame({
        "x": emb_2d[:, 0],
        "y": emb_2d[:, 1],
        "text": texts
    })
    if labels is not None:
        df["label"] = labels
    return df