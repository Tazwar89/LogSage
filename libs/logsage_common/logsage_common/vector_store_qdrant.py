"""
Qdrant-backed replacement for the FAISS + shared-volume VectorStore.

Same public interface (embed/build_index/query/save/load) so ingestion_service
and analysis_service need no logic changes beyond swapping which class they
instantiate. Unlike the FAISS version, there is no local index file: both
services talk to the same Qdrant server, so save()/load() become no-ops --
writes are visible to readers immediately, with no shared PVC/volume needed.
"""
import os, uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


class QdrantVectorStore:
    def __init__(self, collection_name: str, dim: int = 384):
        self.collection_name = collection_name
        self.dim = dim
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = QdrantClient(url=QDRANT_URL)


    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


    def build_index(self, templates: dict):
        """templates: {template_id: template_string} -- recreates the collection fresh each call,
        matching the FAISS version's rebuild-on-baseline-upload behavior."""
        ids = list(templates.keys())
        texts = list(templates.values())

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.EUCLID),
        )

        if not texts:
            return

        vectors = self.embed(texts)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload={"template_id": ids[i], "text": texts[i]},
            )

            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)


    def query(self, text, k=1):
        vec = self.embed([text])[0].tolist()

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vec,
                limit=k,
            )
            hits = response.points

        except UnexpectedResponse:
            return []  # collection doesn't exist yet -- same as "no baseline" in the FAISS version

        return [
            {"template_id": h.payload["template_id"], "text": h.payload["text"], "distance": h.score}
            for h in hits
        ]


    def save(self, path=None):
        pass  # no-op: Qdrant persists server-side on every upsert


    def load(self, path=None):
        """Raises if the collection is missing, matching the FAISS version's
        FileNotFoundError behavior so main.py's existing try/except still works."""
        try:
            self.client.get_collection(self.collection_name)

        except UnexpectedResponse as e:
            raise FileNotFoundError(f"Qdrant collection '{self.collection_name}' does not exist yet") from e