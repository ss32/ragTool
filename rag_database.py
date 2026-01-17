"""
RAG Database Module
Handles ChromaDB operations for storing and retrieving document embeddings.
"""

import asyncio
import os
import sys
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from ollama_utils import ensure_model_available, OllamaModelError

DEFAULT_DB_PATH = os.path.expanduser("~/.rag_tool/chromadb")
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_EMBEDDING_MODEL = "all-minilm:33m"
DEFAULT_EMBEDDING_CACHE_SIZE = 1000


class RAGDatabase:
    """ChromaDB-based RAG database for document storage and retrieval."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_cache_size: int = DEFAULT_EMBEDDING_CACHE_SIZE,
        hnsw_search_ef: int = 100
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Embedding cache (LRU)
        self._embedding_cache: OrderedDict = OrderedDict()
        self._embedding_cache_size = embedding_cache_size
        self._cache_hits = 0
        self._cache_misses = 0

        # Ensure embedding model is available (download if needed)
        try:
            ensure_model_available(embedding_model)
        except OllamaModelError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Ensure database directory exists
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=db_path)

        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # Get or create collection with tuned HNSW parameters
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": hnsw_search_ef,  # Higher = more accurate but slower
            }
        )

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata for ChromaDB compatibility.

        ChromaDB only accepts str, int, float, or bool values.
        Complex types (dict, list) are converted to JSON strings.
        None values are converted to empty strings.
        """
        import json
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (dict, list)):
                # Convert complex types to JSON string
                sanitized[key] = json.dumps(value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
        quiet: bool = False
    ) -> int:
        """Add documents to the database.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process at once
            quiet: If True, suppress progress output

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        total_added = 0

        # Get current count to generate unique IDs
        current_count = self.collection.count()

        # Process in batches for memory efficiency
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in batch]
            # Sanitize metadata to ensure ChromaDB compatibility
            metadatas = [self._sanitize_metadata(doc.metadata) for doc in batch]
            # Use current count + offset to ensure unique IDs across sessions
            ids = [f"doc_{current_count + i + j}" for j in range(len(batch))]

            # Generate embeddings
            if not quiet:
                print(f"Generating embeddings for batch {i // batch_size + 1}...")
            embeddings = self.embeddings.embed_documents(texts)

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            total_added += len(batch)
            if not quiet:
                print(f"  Added {total_added}/{len(documents)} documents")

        return total_added

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cached_embedding(self, text: str) -> list:
        """Get embedding with LRU caching."""
        cache_key = self._get_cache_key(text)

        if cache_key in self._embedding_cache:
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(cache_key)
            self._cache_hits += 1
            return self._embedding_cache[cache_key]

        # Cache miss - generate embedding
        self._cache_misses += 1
        embedding = self.embeddings.embed_query(text)

        # Add to cache
        self._embedding_cache[cache_key] = embedding

        # Evict oldest if at capacity
        while len(self._embedding_cache) > self._embedding_cache_size:
            self._embedding_cache.popitem(last=False)

        return embedding

    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_size': len(self._embedding_cache),
            'max_size': self._embedding_cache_size,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }

    def clear_embedding_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> List[dict]:
        """Search for similar documents."""
        # Generate query embedding (with caching)
        query_embedding = self._get_cached_embedding(query)

        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        return formatted_results

    async def _get_cached_embedding_async(self, text: str) -> list:
        """Get embedding with LRU caching (async version).

        Uses asyncio.to_thread to run the blocking embedding call
        in a thread pool, allowing concurrent operations.
        """
        cache_key = self._get_cache_key(text)

        if cache_key in self._embedding_cache:
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(cache_key)
            self._cache_hits += 1
            return self._embedding_cache[cache_key]

        # Cache miss - generate embedding in thread pool
        self._cache_misses += 1
        embedding = await asyncio.to_thread(self.embeddings.embed_query, text)

        # Add to cache
        self._embedding_cache[cache_key] = embedding

        # Evict oldest if at capacity
        while len(self._embedding_cache) > self._embedding_cache_size:
            self._embedding_cache.popitem(last=False)

        return embedding

    async def search_async(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> List[dict]:
        """Search for similar documents (async version).

        Uses asyncio.to_thread for blocking operations, allowing
        concurrent searches and other async operations.

        Args:
            query: The search query text
            n_results: Number of results to return
            where: Optional filter dictionary for ChromaDB

        Returns:
            List of result dictionaries with content, metadata, and distance
        """
        # Generate query embedding (with caching) - async
        query_embedding = await self._get_cached_embedding_async(query)

        # Search in collection - run in thread pool
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        return formatted_results

    def get_stats(self) -> dict:
        """Get database statistics."""
        return {
            'collection_name': self.collection_name,
            'document_count': self.collection.count(),
            'db_path': self.db_path,
            'embedding_model': self.embedding_model
        }

    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection: {self.collection_name}")

    def delete_database(self):
        """Delete the entire database."""
        import shutil
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"Deleted database at: {self.db_path}")


def create_database(
    documents: List[Document],
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    clear_existing: bool = True
) -> RAGDatabase:
    """Create a new RAG database from documents."""
    db = RAGDatabase(db_path, collection_name, embedding_model)

    if clear_existing:
        db.clear()

    print(f"\nCreating RAG database...")
    print(f"  Database path: {db_path}")
    print(f"  Collection: {collection_name}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Documents to add: {len(documents)}")

    added = db.add_documents(documents)
    print(f"\nDatabase created with {added} document chunks")

    return db


if __name__ == "__main__":
    # Test the database
    from document_loader import load_documents, chunk_documents

    import sys
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        test_dir = "/tmp/inbounds"

    # Load and chunk documents
    docs = load_documents(test_dir)
    chunks = chunk_documents(docs)

    # Create database
    db = create_database(chunks)

    # Test search
    print("\n--- Testing Search ---")
    results = db.search("What is the purpose of this code?", n_results=3)
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"  Source: {result['metadata'].get('source', 'unknown')}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Preview: {result['content'][:150]}...")
