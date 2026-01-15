"""
RAG Database Module
Handles ChromaDB operations for storing and retrieving document embeddings.
"""

import os
import sys
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


class RAGDatabase:
    """ChromaDB-based RAG database for document storage and retrieval."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model

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

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> int:
        """Add documents to the database."""
        if not documents:
            return 0

        total_added = 0

        # Process in batches for memory efficiency
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            ids = [f"doc_{i + j}" for j in range(len(batch))]

            # Generate embeddings
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
            print(f"  Added {total_added}/{len(documents)} documents")

        return total_added

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> List[dict]:
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

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
