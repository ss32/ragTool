"""
Query Engine Module
Handles RAG-based querying with LLM response generation.
"""

import asyncio
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Iterator, AsyncIterator
from langchain_ollama import ChatOllama
from rag_database import RAGDatabase, DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
from ollama_utils import ensure_model_available, OllamaModelError

DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_RESPONSE_CACHE_SIZE = 500


class QueryEngine:
    """RAG-based query engine with LLM response generation."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        warm_up: bool = True,
        enable_response_cache: bool = True,
        response_cache_size: int = DEFAULT_RESPONSE_CACHE_SIZE,
        hnsw_search_ef: int = 100
    ):
        self.db = RAGDatabase(db_path, collection_name, embedding_model, hnsw_search_ef=hnsw_search_ef)

        # Ensure LLM model is available (download if needed)
        try:
            ensure_model_available(llm_model)
        except OllamaModelError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        self.llm = ChatOllama(model=llm_model, temperature=0.1)
        self.llm_model = llm_model

        # Response cache
        self._enable_response_cache = enable_response_cache
        self._response_cache_size = response_cache_size
        self._response_cache = {}
        self._response_cache_hits = 0
        self._response_cache_misses = 0
        self._cache_file = Path(db_path) / "response_cache.json"
        if enable_response_cache:
            self._load_response_cache()

        # Warm up the LLM model to load it into memory
        if warm_up:
            self._warm_up_model()

    def _warm_up_model(self):
        """Send a minimal prompt to load the LLM into GPU memory."""
        try:
            self.llm.invoke("Hi")
        except Exception:
            pass  # Ignore warm-up errors

    def _load_response_cache(self):
        """Load response cache from disk."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    self._response_cache = json.load(f)
            except Exception:
                self._response_cache = {}

    def _save_response_cache(self):
        """Save response cache to disk."""
        try:
            # Ensure directory exists
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'w') as f:
                json.dump(self._response_cache, f)
        except Exception:
            pass  # Ignore cache save errors

    def _response_cache_key(self, question: str, context: str) -> str:
        """Generate a cache key from question and context."""
        content = f"{question}|{context}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def clear_response_cache(self):
        """Clear the response cache."""
        self._response_cache = {}
        self._response_cache_hits = 0
        self._response_cache_misses = 0
        if self._cache_file.exists():
            self._cache_file.unlink()

    def get_cache_stats(self) -> dict:
        """Get response cache statistics."""
        total = self._response_cache_hits + self._response_cache_misses
        hit_rate = (self._response_cache_hits / total * 100) if total > 0 else 0
        return {
            'response_cache_size': len(self._response_cache),
            'response_cache_max': self._response_cache_size,
            'response_cache_hits': self._response_cache_hits,
            'response_cache_misses': self._response_cache_misses,
            'response_cache_hit_rate': f"{hit_rate:.1f}%",
            **self.db.get_cache_stats()
        }

    def _build_context_and_sources(self, results: List[dict]) -> tuple:
        """Build context string and source list from search results."""
        context_parts = []
        sources = []
        for i, result in enumerate(results):
            context_parts.append(f"[Document {i + 1}]\n{result['content']}")
            source = result['metadata'].get('source', 'unknown')
            if source not in sources:
                sources.append(source)
        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the LLM prompt."""
        return f"""You are a helpful assistant answering questions based on the provided context documents.
Use ONLY the information from the context below to answer the question.
If the context doesn't contain enough information to answer, say so.
Be concise and direct in your response.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    def query(
        self,
        question: str,
        n_results: int = 5,
        show_sources: bool = True
    ) -> str:
        """Query the RAG database and generate a response."""
        # Search for relevant documents
        results = self.db.search(question, n_results=n_results)

        if not results:
            return "No relevant documents found in the database."

        # Build context from search results
        context, sources = self._build_context_and_sources(results)

        # Check response cache
        if self._enable_response_cache:
            cache_key = self._response_cache_key(question, context)
            if cache_key in self._response_cache:
                self._response_cache_hits += 1
                answer = self._response_cache[cache_key]
                if show_sources and sources:
                    source_list = "\n".join(f"  - {s}" for s in sources)
                    answer += f"\n\nSources:\n{source_list}"
                return answer
            self._response_cache_misses += 1

        # Build prompt and generate response
        prompt = self._build_prompt(question, context)
        response = self.llm.invoke(prompt)
        answer = response.content

        # Cache the response (without sources, they're added dynamically)
        if self._enable_response_cache:
            self._response_cache[cache_key] = answer
            # Limit cache size
            while len(self._response_cache) > self._response_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            self._save_response_cache()

        # Add sources if requested
        if show_sources and sources:
            source_list = "\n".join(f"  - {s}" for s in sources)
            answer += f"\n\nSources:\n{source_list}"

        return answer

    def query_stream(
        self,
        question: str,
        n_results: int = 5,
        show_sources: bool = True
    ) -> Iterator[str]:
        """Query with streaming response for faster perceived latency."""
        # Search for relevant documents
        results = self.db.search(question, n_results=n_results)

        if not results:
            yield "No relevant documents found in the database."
            return

        # Build context from search results
        context, sources = self._build_context_and_sources(results)

        # Check response cache first
        if self._enable_response_cache:
            cache_key = self._response_cache_key(question, context)
            if cache_key in self._response_cache:
                self._response_cache_hits += 1
                yield self._response_cache[cache_key]
                if show_sources and sources:
                    source_list = "\n".join(f"  - {s}" for s in sources)
                    yield f"\n\nSources:\n{source_list}"
                return
            self._response_cache_misses += 1

        # Build prompt and stream response
        prompt = self._build_prompt(question, context)
        full_response = []

        for chunk in self.llm.stream(prompt):
            content = chunk.content
            full_response.append(content)
            yield content

        # Cache the complete response
        answer = ''.join(full_response)
        if self._enable_response_cache:
            self._response_cache[cache_key] = answer
            while len(self._response_cache) > self._response_cache_size:
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            self._save_response_cache()

        # Add sources at the end
        if show_sources and sources:
            source_list = "\n".join(f"  - {s}" for s in sources)
            yield f"\n\nSources:\n{source_list}"

    def search_only(self, query: str, n_results: int = 5) -> List[dict]:
        """Perform similarity search without LLM generation."""
        return self.db.search(query, n_results=n_results)

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def _warm_up_model_async(self):
        """Send a minimal prompt to load the LLM into GPU memory (async version)."""
        try:
            await asyncio.to_thread(self.llm.invoke, "Hi")
        except Exception:
            pass  # Ignore warm-up errors

    async def search_only_async(self, query: str, n_results: int = 5) -> List[dict]:
        """Perform similarity search without LLM generation (async version)."""
        return await self.db.search_async(query, n_results=n_results)

    async def query_async(
        self,
        question: str,
        n_results: int = 5,
        show_sources: bool = True
    ) -> str:
        """Query the RAG database and generate a response (async version).

        Uses asyncio.to_thread for blocking LLM operations, allowing
        concurrent queries and other async operations.

        Args:
            question: The question to answer
            n_results: Number of documents to retrieve for context
            show_sources: Whether to include source documents in the response

        Returns:
            The generated answer string
        """
        # Search for relevant documents (async)
        results = await self.db.search_async(question, n_results=n_results)

        if not results:
            return "No relevant documents found in the database."

        # Build context from search results
        context, sources = self._build_context_and_sources(results)

        # Check response cache
        if self._enable_response_cache:
            cache_key = self._response_cache_key(question, context)
            if cache_key in self._response_cache:
                self._response_cache_hits += 1
                answer = self._response_cache[cache_key]
                if show_sources and sources:
                    source_list = "\n".join(f"  - {s}" for s in sources)
                    answer += f"\n\nSources:\n{source_list}"
                return answer
            self._response_cache_misses += 1

        # Build prompt and generate response (in thread pool)
        prompt = self._build_prompt(question, context)
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        answer = response.content

        # Cache the response (without sources, they're added dynamically)
        if self._enable_response_cache:
            self._response_cache[cache_key] = answer
            # Limit cache size
            while len(self._response_cache) > self._response_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            self._save_response_cache()

        # Add sources if requested
        if show_sources and sources:
            source_list = "\n".join(f"  - {s}" for s in sources)
            answer += f"\n\nSources:\n{source_list}"

        return answer

    async def query_stream_async(
        self,
        question: str,
        n_results: int = 5,
        show_sources: bool = True
    ) -> AsyncIterator[str]:
        """Query with streaming response (async generator version).

        Uses async iteration for non-blocking streaming of LLM responses.

        Args:
            question: The question to answer
            n_results: Number of documents to retrieve for context
            show_sources: Whether to include source documents at the end

        Yields:
            Response tokens as they are generated
        """
        # Search for relevant documents (async)
        results = await self.db.search_async(question, n_results=n_results)

        if not results:
            yield "No relevant documents found in the database."
            return

        # Build context from search results
        context, sources = self._build_context_and_sources(results)

        # Check response cache first
        if self._enable_response_cache:
            cache_key = self._response_cache_key(question, context)
            if cache_key in self._response_cache:
                self._response_cache_hits += 1
                yield self._response_cache[cache_key]
                if show_sources and sources:
                    source_list = "\n".join(f"  - {s}" for s in sources)
                    yield f"\n\nSources:\n{source_list}"
                return
            self._response_cache_misses += 1

        # Build prompt and stream response
        prompt = self._build_prompt(question, context)
        full_response = []

        # Stream LLM response using thread pool for each chunk
        # We use a queue-based approach to bridge sync streaming with async
        async def stream_llm():
            """Run LLM streaming in thread and yield chunks."""
            import queue
            import threading

            chunk_queue = queue.Queue()
            error_holder = [None]

            def run_stream():
                try:
                    for chunk in self.llm.stream(prompt):
                        chunk_queue.put(chunk.content)
                    chunk_queue.put(None)  # Signal completion
                except Exception as e:
                    error_holder[0] = e
                    chunk_queue.put(None)

            thread = threading.Thread(target=run_stream, daemon=True)
            thread.start()

            while True:
                # Use asyncio.to_thread to avoid blocking
                chunk = await asyncio.to_thread(chunk_queue.get)
                if chunk is None:
                    if error_holder[0]:
                        raise error_holder[0]
                    break
                yield chunk

        async for content in stream_llm():
            full_response.append(content)
            yield content

        # Cache the complete response
        answer = ''.join(full_response)
        if self._enable_response_cache:
            self._response_cache[cache_key] = answer
            while len(self._response_cache) > self._response_cache_size:
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            self._save_response_cache()

        # Add sources at the end
        if show_sources and sources:
            source_list = "\n".join(f"  - {s}" for s in sources)
            yield f"\n\nSources:\n{source_list}"

    async def interactive_session_async(self, use_streaming: bool = True):
        """Start an interactive query session (async version).

        Uses async methods for non-blocking query processing.
        """
        print("\n" + "=" * 60)
        print("RAG Query Interface (Async)")
        print("=" * 60)
        print(f"Database: {self.db.collection_name}")
        print(f"Documents: {self.db.get_stats()['document_count']} chunks")
        print(f"LLM Model: {self.llm_model}")
        print(f"Streaming: {'enabled' if use_streaming else 'disabled'}")
        print(f"Response Cache: {'enabled' if self._enable_response_cache else 'disabled'}")
        print("-" * 60)
        print("Commands:")
        print("  /search <query>  - Show raw search results")
        print("  /stats           - Show database statistics")
        print("  /cache           - Show cache statistics")
        print("  /clear-cache     - Clear all caches")
        print("  /quit or /exit   - Exit the session")
        print("=" * 60)

        while True:
            try:
                # Use asyncio-compatible input
                user_input = await asyncio.to_thread(input, "\nQuery> ")
                user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("Goodbye!")
                    break

                if user_input.lower() == '/stats':
                    stats = self.db.get_stats()
                    print(f"\nDatabase Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                if user_input.lower() == '/cache':
                    stats = self.get_cache_stats()
                    print(f"\nCache Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                if user_input.lower() == '/clear-cache':
                    self.db.clear_embedding_cache()
                    self.clear_response_cache()
                    print("Caches cleared.")
                    continue

                if user_input.lower().startswith('/search '):
                    query = user_input[8:].strip()
                    results = await self.search_only_async(query, n_results=5)
                    print(f"\nSearch results for: {query}")
                    for i, result in enumerate(results):
                        print(f"\n[{i + 1}] Distance: {result['distance']:.4f}")
                        print(f"    Source: {result['metadata'].get('source', 'unknown')}")
                        print(f"    Preview: {result['content'][:200]}...")
                    continue

                # Regular query
                print("\nSearching and generating response...\n")

                if use_streaming:
                    async for token in self.query_stream_async(user_input):
                        print(token, end='', flush=True)
                    print()
                else:
                    answer = await self.query_async(user_input)
                    print(answer)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def interactive_session(self, use_streaming: bool = True):
        """Start an interactive query session."""
        print("\n" + "=" * 60)
        print("RAG Query Interface")
        print("=" * 60)
        print(f"Database: {self.db.collection_name}")
        print(f"Documents: {self.db.get_stats()['document_count']} chunks")
        print(f"LLM Model: {self.llm_model}")
        print(f"Streaming: {'enabled' if use_streaming else 'disabled'}")
        print(f"Response Cache: {'enabled' if self._enable_response_cache else 'disabled'}")
        print("-" * 60)
        print("Commands:")
        print("  /search <query>  - Show raw search results")
        print("  /stats           - Show database statistics")
        print("  /cache           - Show cache statistics")
        print("  /clear-cache     - Clear all caches")
        print("  /quit or /exit   - Exit the session")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nQuery> ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("Goodbye!")
                    break

                if user_input.lower() == '/stats':
                    stats = self.db.get_stats()
                    print(f"\nDatabase Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                if user_input.lower() == '/cache':
                    stats = self.get_cache_stats()
                    print(f"\nCache Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                if user_input.lower() == '/clear-cache':
                    self.db.clear_embedding_cache()
                    self.clear_response_cache()
                    print("Caches cleared.")
                    continue

                if user_input.lower().startswith('/search '):
                    query = user_input[8:].strip()
                    results = self.search_only(query, n_results=5)
                    print(f"\nSearch results for: {query}")
                    for i, result in enumerate(results):
                        print(f"\n[{i + 1}] Distance: {result['distance']:.4f}")
                        print(f"    Source: {result['metadata'].get('source', 'unknown')}")
                        print(f"    Preview: {result['content'][:200]}...")
                    continue

                # Regular query
                print("\nSearching and generating response...\n")

                if use_streaming:
                    for token in self.query_stream(user_input):
                        print(token, end='', flush=True)
                    print()
                else:
                    answer = self.query(user_input)
                    print(answer)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


if __name__ == "__main__":
    import sys

    engine = QueryEngine()

    if len(sys.argv) > 1:
        # Single query mode
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}")
        print("-" * 40)
        answer = engine.query(question)
        print(answer)
    else:
        # Interactive mode
        engine.interactive_session()
