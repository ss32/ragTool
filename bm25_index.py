"""
BM25 Index Module
Handles BM25 keyword-based search index creation and retrieval.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import bm25s
import Stemmer


class BM25Index:
    """BM25 index manager for keyword-based document retrieval."""

    def __init__(
        self,
        index_path: str,
        stemmer_language: str = "english"
    ):
        """Initialize BM25 index.

        Args:
            index_path: Directory to store BM25 index files
            stemmer_language: Language for stemming (default: english)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / "bm25_index.pkl"
        self.corpus_file = self.index_path / "corpus.json"
        self.stemmer_language = stemmer_language

        # Initialize stemmer for tokenization
        self.stemmer = Stemmer.Stemmer(stemmer_language)

        # BM25 index and document mapping
        self.retriever: Optional[bm25s.BM25] = None
        self.doc_ids: List[str] = []  # Maps index position to document ID
        self.doc_contents: List[str] = []  # Original document contents

        # Load existing index if available
        self._load_index()

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts with stemming and stopword removal.

        Args:
            texts: List of text strings to tokenize

        Returns:
            List of tokenized documents (list of token lists)
        """
        # Use bm25s built-in tokenizer with stemming
        tokens = bm25s.tokenize(
            texts,
            stemmer=self.stemmer,
            stopwords="en"
        )
        return tokens

    def _load_index(self):
        """Load existing index from disk if available."""
        if self.index_file.exists() and self.corpus_file.exists():
            try:
                # Load corpus mapping
                with open(self.corpus_file, 'r') as f:
                    corpus_data = json.load(f)
                    self.doc_ids = corpus_data.get('doc_ids', [])
                    self.doc_contents = corpus_data.get('doc_contents', [])

                # Load BM25 index
                with open(self.index_file, 'rb') as f:
                    self.retriever = pickle.load(f)

            except Exception as e:
                print(f"Warning: Failed to load BM25 index: {e}")
                self.retriever = None
                self.doc_ids = []
                self.doc_contents = []

    def _save_index(self):
        """Save index to disk."""
        if self.retriever is None:
            return

        try:
            # Save corpus mapping
            with open(self.corpus_file, 'w') as f:
                json.dump({
                    'doc_ids': self.doc_ids,
                    'doc_contents': self.doc_contents
                }, f)

            # Save BM25 index
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.retriever, f)

        except Exception as e:
            print(f"Warning: Failed to save BM25 index: {e}")

    def add_documents(
        self,
        documents: List[Dict],
        doc_ids: List[str],
        quiet: bool = False
    ) -> int:
        """Add documents to the BM25 index.

        Args:
            documents: List of document dicts with 'content' key
            doc_ids: List of document IDs corresponding to each document
            quiet: If True, suppress progress output

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Extract content from documents
        new_contents = [doc.get('content', '') or doc.get('page_content', '') for doc in documents]

        # If we have existing documents, we need to rebuild the index
        # (bm25s doesn't support incremental updates)
        all_contents = self.doc_contents + new_contents
        all_ids = self.doc_ids + doc_ids

        if not quiet:
            print(f"Building BM25 index with {len(all_contents)} documents...")

        # Tokenize all documents
        tokens = self._tokenize(all_contents)

        # Create new BM25 index
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)

        # Update document mappings
        self.doc_ids = all_ids
        self.doc_contents = all_contents

        # Save to disk
        self._save_index()

        return len(new_contents)

    def search(
        self,
        query: str,
        n_results: int = 10
    ) -> List[Dict]:
        """Search for documents using BM25.

        Args:
            query: Search query string
            n_results: Number of results to return

        Returns:
            List of result dicts with 'doc_id', 'content', 'score', 'rank'
        """
        if self.retriever is None or not self.doc_ids:
            return []

        # Tokenize query
        query_tokens = self._tokenize([query])

        # Limit n_results to available documents
        n_results = min(n_results, len(self.doc_ids))

        # Search
        results, scores = self.retriever.retrieve(query_tokens, k=n_results)

        # Format results
        formatted_results = []
        for rank, (idx, score) in enumerate(zip(results[0], scores[0])):
            if idx < len(self.doc_ids):
                formatted_results.append({
                    'doc_id': self.doc_ids[idx],
                    'content': self.doc_contents[idx],
                    'bm25_score': float(score),
                    'bm25_rank': rank
                })

        return formatted_results

    def get_stats(self) -> Dict:
        """Get index statistics.

        Returns:
            Dict with index statistics
        """
        return {
            'document_count': len(self.doc_ids),
            'index_path': str(self.index_path),
            'has_index': self.retriever is not None
        }

    def clear(self):
        """Clear the index."""
        self.retriever = None
        self.doc_ids = []
        self.doc_contents = []

        # Remove index files
        if self.index_file.exists():
            self.index_file.unlink()
        if self.corpus_file.exists():
            self.corpus_file.unlink()

    def delete(self):
        """Delete the entire index directory."""
        import shutil
        if self.index_path.exists():
            shutil.rmtree(self.index_path)


def create_bm25_index(
    documents: List[Dict],
    index_path: str,
    doc_ids: Optional[List[str]] = None
) -> BM25Index:
    """Create a new BM25 index from documents.

    Args:
        documents: List of document dicts with 'content' key
        index_path: Directory to store the index
        doc_ids: Optional list of document IDs (auto-generated if not provided)

    Returns:
        BM25Index instance
    """
    index = BM25Index(index_path)
    index.clear()  # Start fresh

    # Generate doc IDs if not provided
    if doc_ids is None:
        doc_ids = [f"doc_{i}" for i in range(len(documents))]

    index.add_documents(documents, doc_ids)
    return index


if __name__ == "__main__":
    # Test the BM25 index
    test_docs = [
        {'content': 'The quick brown fox jumps over the lazy dog'},
        {'content': 'A quick brown dog runs in the park'},
        {'content': 'Python is a programming language'},
        {'content': 'Machine learning uses algorithms to learn from data'},
        {'content': 'The fox and the dog became friends'},
    ]

    # Create index
    index = create_bm25_index(test_docs, "/tmp/test_bm25")

    print("Index stats:", index.get_stats())

    # Test search
    query = "quick brown fox"
    results = index.search(query, n_results=3)

    print(f"\nSearch results for: '{query}'")
    for r in results:
        print(f"  [{r['bm25_rank']}] Score: {r['bm25_score']:.4f}")
        print(f"      Content: {r['content'][:50]}...")
