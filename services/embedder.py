from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Set device to CPU and optimize memory usage
        device = "cpu"
        # Configure lower memory usage
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)  # Limit CPU threads

        # Load model with memory-efficient settings
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[List[float]]:
        """Embed texts with memory-efficient batching."""
        if isinstance(texts, str):
            texts = [texts]

        # For small number of texts, process directly
        if len(texts) <= batch_size:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

        # For larger sets, use batching to avoid memory issues
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings.tolist())

        return all_embeddings

if __name__ == "__main__":
    embedder = Embedder()
    sample_texts = [
        "Linux provides powerful command-line tools.",
        "Python is a high-level programming language."
    ]
    embeddings = embedder.embed(sample_texts)
    for i, emb in enumerate(embeddings):
        print(f"Text {i+1}: {sample_texts[i]}")
        print(f"Embedding (first 5 dims): {emb[:5]}")
        print(f"Embedding length: {len(emb)}\n")
