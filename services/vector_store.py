from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
import numpy as np

class QdrantVectorStore:
    def __init__(self, host: str = "host.docker.internal", port: int = 6333, collection_name: str = "docs", vector_size: int = 384):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def upsert(self, vectors: List[List[float]], payloads: Optional[List[Dict[str, Any]]] = None):
        points = []
        for idx, vector in enumerate(vectors):
            payload = payloads[idx] if payloads else {}
            points.append(PointStruct(id=idx, vector=vector, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        results = self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=limit)
        return [{"score": r.score, "payload": r.payload} for r in results]

if __name__ == "__main__":
    # Dummy test: insert and search random vectors
    store = QdrantVectorStore(host="host.docker.internal", collection_name="test_docs", vector_size=5)
    np.random.seed(42)
    dummy_vectors = np.random.rand(3, 5).tolist()
    payloads = [
        {"text": "Linux provides powerful command-line tools."},
        {"text": "Python is a high-level programming language."},
        {"text": "Git is a distributed version control system."}
    ]
    store.upsert(dummy_vectors, payloads)
    print("Inserted 3 dummy vectors.")
    query = dummy_vectors[0]
    results = store.search(query, limit=2)
    print("Search results:")
    for r in results:
        print(r)

    # Example: Insert and search a real embedding for a text chunk
    from embedder import Embedder
    embedder = Embedder()
    text_chunk = "Linux provides powerful command-line tools for system administration and development."
    embedding = embedder.embed([text_chunk])[0]
    store = QdrantVectorStore(host="host.docker.internal", collection_name="test_docs", vector_size=len(embedding))
    payload = {"text": text_chunk}
    store.upsert([embedding], [payload])
    print("Inserted 1 embedding for test chunk.")
    results = store.search(embedding, limit=1)
    print("Search results:")
    for r in results:
        print(r)
