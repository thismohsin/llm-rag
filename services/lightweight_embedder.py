"""
Ultra lightweight embedder for testing on memory-constrained environments.
Uses a basic TF-IDF vectorization approach instead of neural network models.
"""

from typing import List, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class LightweightEmbedder:
    def __init__(self, max_features=384):
        """Initialize a lightweight TF-IDF based embedder.

        Args:
            max_features: Output embedding dimension
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            norm='l2'  # L2 norm for cosine similarity
        )
        # Initialize with some common words to build vocabulary
        init_texts = [
            "the quick brown fox jumps over the lazy dog",
            "hello world example text for embedding testing",
            "linux python git programming computer technology"
        ]
        self.vectorizer.fit(init_texts)

    def embed(self, texts: Union[str, List[str]], batch_size=None) -> List[List[float]]:
        """Convert texts to embeddings using TF-IDF.

        Args:
            texts: String or list of strings to embed
            batch_size: Ignored, included for API compatibility

        Returns:
            List of embeddings (list of floats)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Get sparse matrix
        sparse_embeddings = self.vectorizer.transform(texts)

        # Convert to dense numpy arrays and then to Python lists
        dense_embeddings = sparse_embeddings.toarray()

        # Pad or truncate to ensure consistent dimensions
        result = []
        for emb in dense_embeddings:
            if len(emb) < self.max_features:
                # Pad with zeros
                emb = np.pad(emb, (0, self.max_features - len(emb)))
            elif len(emb) > self.max_features:
                # Truncate
                emb = emb[:self.max_features]
            result.append(emb.tolist())

        return result

if __name__ == "__main__":
    # Quick test
    embedder = LightweightEmbedder(max_features=384)
    texts = ["This is a test", "Another example sentence"]
    embeddings = embedder.embed(texts)

    for i, emb in enumerate(embeddings):
        print(f"Text: {texts[i]}")
        print(f"Embedding size: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
        print()
