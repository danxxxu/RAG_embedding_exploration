import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = np.array(embeddings)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(query_embedding, k)
        return indices[0]