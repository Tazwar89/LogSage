import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = os.environ.get("VECTOR_INDEX_PATH", "index")

class VectorStore:
    def __init__(self, dim=384):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatL2(dim)
        self.id_map = {}  # faiss internal index -> metadata


    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


    def build_index(self, templates: dict):
        """templates: {template_id: template_string}"""
        ids = list(templates.keys())
        texts = list(templates.values())
        self.index = faiss.IndexFlatL2(self.index.d)

        if texts:
            vectors = self.embed(texts)
            self.index.add(np.array(vectors, dtype="float32"))

        self.id_map = {i: {"template_id": ids[i], "text": texts[i]} for i in range(len(ids))}


    def query(self, text, k=1):
        vec = self.embed([text]).astype("float32")
        distances, indices = self.index.search(vec, k)
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            results.append({**self.id_map[idx], "distance": float(dist)})

        return results


    def save(self, path=None):
        path = path or DEFAULT_INDEX_PATH
        faiss.write_index(self.index, f"{path}.faiss")

        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump(self.id_map, f)


    def load(self, path=None):
        path = path or DEFAULT_INDEX_PATH
        self.index = faiss.read_index(f"{path}.faiss")

        with open(f"{path}_meta.pkl", "rb") as f:
            self.id_map = pickle.load(f)