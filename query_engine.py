"""
Query Engine Module
Handles RAG-based querying with LLM response generation.
"""

import sys
from typing import List, Optional
from langchain_ollama import ChatOllama
from rag_database import RAGDatabase, DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
from ollama_utils import ensure_model_available, OllamaModelError

DEFAULT_LLM_MODEL = "qwen2.5:7b"


class QueryEngine:
    """RAG-based query engine with LLM response generation."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL
    ):
        self.db = RAGDatabase(db_path, collection_name, embedding_model)

        # Ensure LLM model is available (download if needed)
        try:
            ensure_model_available(llm_model)
        except OllamaModelError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        self.llm = ChatOllama(model=llm_model, temperature=0.1)
        self.llm_model = llm_model

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
        context_parts = []
        sources = []
        for i, result in enumerate(results):
            context_parts.append(f"[Document {i + 1}]\n{result['content']}")
            source = result['metadata'].get('source', 'unknown')
            if source not in sources:
                sources.append(source)

        context = "\n\n---\n\n".join(context_parts)

        # Build prompt
        prompt = f"""You are a helpful assistant answering questions based on the provided context documents.
Use ONLY the information from the context below to answer the question.
If the context doesn't contain enough information to answer, say so.
Be concise and direct in your response.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        # Generate response using LLM
        response = self.llm.invoke(prompt)
        answer = response.content

        # Add sources if requested
        if show_sources and sources:
            source_list = "\n".join(f"  - {s}" for s in sources)
            answer += f"\n\nSources:\n{source_list}"

        return answer

    def search_only(self, query: str, n_results: int = 5) -> List[dict]:
        """Perform similarity search without LLM generation."""
        return self.db.search(query, n_results=n_results)

    def interactive_session(self):
        """Start an interactive query session."""
        print("\n" + "=" * 60)
        print("RAG Query Interface")
        print("=" * 60)
        print(f"Database: {self.db.collection_name}")
        print(f"Documents: {self.db.get_stats()['document_count']} chunks")
        print(f"LLM Model: {self.llm_model}")
        print("-" * 60)
        print("Commands:")
        print("  /search <query>  - Show raw search results")
        print("  /stats           - Show database statistics")
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
                print("\nSearching and generating response...")
                answer = self.query(user_input)
                print(f"\n{answer}")

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
