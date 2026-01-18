"""
Reranker Module
Cross-encoder reranking for improved retrieval quality.
"""

import sys
from typing import List, Dict, Optional

# Default model - good balance of speed and quality
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Available reranker models with their characteristics
RERANKER_MODELS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "description": "Fast, good quality (default)",
        "speed": "fast",
        "quality": "good"
    },
    "cross-encoder/ms-marco-MiniLM-L-12-v2": {
        "description": "Medium speed, better quality",
        "speed": "medium",
        "quality": "better"
    },
    "BAAI/bge-reranker-v2-m3": {
        "description": "Slow, best quality, multilingual",
        "speed": "slow",
        "quality": "best"
    }
}


class Reranker:
    """Cross-encoder reranker for improving retrieval results."""

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        device: Optional[str] = None
    ):
        """Initialize the reranker.

        Args:
            model: Cross-encoder model name (see RERANKER_MODELS for options)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model
        self._model = None
        self._device = device

    def _load_model(self):
        """Lazily load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                print(
                    "Error: sentence-transformers is required for reranking.\n"
                    "Install with: pip install sentence-transformers",
                    file=sys.stderr
                )
                raise

            print(f"Loading reranker model: {self.model_name}...")
            self._model = CrossEncoder(self.model_name, device=self._device)
            print(f"Reranker model loaded.")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        content_key: str = 'content'
    ) -> List[Dict]:
        """Rerank documents using cross-encoder scores.

        Args:
            query: The search query
            documents: List of document dicts containing content to rerank
            top_k: Number of top results to return (None = return all)
            content_key: Key in document dict containing the text content

        Returns:
            Reranked list of documents with added 'rerank_score' field
        """
        if not documents:
            return []

        self._load_model()

        # Prepare query-document pairs
        pairs = []
        valid_indices = []
        for i, doc in enumerate(documents):
            content = doc.get(content_key, '')
            if content:
                pairs.append([query, content])
                valid_indices.append(i)

        if not pairs:
            return documents

        # Get cross-encoder scores
        scores = self._model.predict(pairs)

        # Create results with scores
        results = []
        for idx, score in zip(valid_indices, scores):
            doc = documents[idx].copy()
            doc['rerank_score'] = float(score)
            results.append(doc)

        # Sort by rerank score (descending)
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def get_model_info(self) -> Dict:
        """Get information about the current reranker model.

        Returns:
            Dict with model information
        """
        info = RERANKER_MODELS.get(self.model_name, {
            "description": "Custom model",
            "speed": "unknown",
            "quality": "unknown"
        })
        return {
            "model": self.model_name,
            "loaded": self._model is not None,
            **info
        }


# Singleton instance for reuse
_reranker_instance: Optional[Reranker] = None


def get_reranker(
    model: str = DEFAULT_RERANK_MODEL,
    device: Optional[str] = None
) -> Reranker:
    """Get a reranker instance (singleton pattern for efficiency).

    Args:
        model: Cross-encoder model name
        device: Device to use

    Returns:
        Reranker instance
    """
    global _reranker_instance

    # Create new instance if model changed or not initialized
    if _reranker_instance is None or _reranker_instance.model_name != model:
        _reranker_instance = Reranker(model=model, device=device)

    return _reranker_instance


def rerank_results(
    query: str,
    documents: List[Dict],
    top_k: Optional[int] = None,
    model: str = DEFAULT_RERANK_MODEL
) -> List[Dict]:
    """Convenience function to rerank results.

    Args:
        query: The search query
        documents: List of document dicts to rerank
        top_k: Number of results to return
        model: Reranker model to use

    Returns:
        Reranked documents
    """
    reranker = get_reranker(model=model)
    return reranker.rerank(query, documents, top_k=top_k)


if __name__ == "__main__":
    # Test the reranker
    test_docs = [
        {'content': 'Python is a programming language used for web development.'},
        {'content': 'The quick brown fox jumps over the lazy dog.'},
        {'content': 'Python programming enables data science and machine learning.'},
        {'content': 'Machine learning is a subset of artificial intelligence.'},
        {'content': 'Python was created by Guido van Rossum in 1991.'},
    ]

    query = "What is Python used for?"

    print(f"Query: {query}")
    print("\nOriginal order:")
    for i, doc in enumerate(test_docs):
        print(f"  [{i + 1}] {doc['content'][:60]}...")

    # Rerank
    reranked = rerank_results(query, test_docs, top_k=3)

    print("\nReranked order (top 3):")
    for i, doc in enumerate(reranked):
        print(f"  [{i + 1}] Score: {doc['rerank_score']:.4f}")
        print(f"       {doc['content'][:60]}...")
