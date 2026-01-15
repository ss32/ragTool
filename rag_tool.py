#!/usr/bin/env python3
"""
RAG Tool - Document Database Creation and Query Tool

A command-line tool for creating and querying RAG (Retrieval-Augmented Generation)
databases from local documents.

Usage:
    python rag_tool.py create --input /path/to/docs
    python rag_tool.py query "What does this code do?"
    python rag_tool.py interactive
"""

import argparse
import sys
import os

from document_loader import load_documents, chunk_documents
from rag_database import (
    create_database, RAGDatabase,
    DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
)
from query_engine import QueryEngine, DEFAULT_LLM_MODEL
from code_summarizer import summarize_code_documents, DEFAULT_SUMMARIZER_MODEL


def cmd_create(args):
    """Create a new RAG database from documents."""
    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    print(f"Creating RAG database from: {input_dir}")
    print("=" * 60)

    # Load documents
    docs = load_documents(input_dir, recursive=not args.no_recursive)

    if not docs:
        print("No supported documents found in the directory.")
        sys.exit(1)

    # Summarize code documents if requested
    if args.summarize:
        print("\nGenerating documentation for code files...")
        docs = summarize_code_documents(
            docs,
            model=args.summarize_model,
            show_progress=True
        )

    # Chunk documents
    chunks = chunk_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Create database
    db = create_database(
        chunks,
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        clear_existing=not args.append
    )

    print("\n" + "=" * 60)
    print("Database creation complete!")
    stats = db.get_stats()
    print(f"  Total chunks: {stats['document_count']}")
    print(f"  Database path: {stats['db_path']}")
    print(f"  Collection: {stats['collection_name']}")
    print("\nTo query: python rag_tool.py query \"your question here\"")
    print("For interactive mode: python rag_tool.py interactive")


def cmd_query(args):
    """Query the RAG database."""
    engine = QueryEngine(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model
    )

    stats = engine.db.get_stats()
    if stats['document_count'] == 0:
        print("Error: Database is empty. Run 'create' first.")
        sys.exit(1)

    question = args.question
    print(f"Question: {question}")
    print("-" * 60)

    if args.search_only:
        results = engine.search_only(question, n_results=args.results)
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n[{i + 1}] Distance: {result['distance']:.4f}")
            print(f"    Source: {result['metadata'].get('source', 'unknown')}")
            print(f"    Content: {result['content'][:300]}...")
    else:
        answer = engine.query(
            question,
            n_results=args.results,
            show_sources=not args.no_sources
        )
        print(answer)


def cmd_interactive(args):
    """Start interactive query session."""
    engine = QueryEngine(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model
    )

    stats = engine.db.get_stats()
    if stats['document_count'] == 0:
        print("Error: Database is empty. Run 'create' first.")
        sys.exit(1)

    engine.interactive_session()


def cmd_stats(args):
    """Show database statistics."""
    db = RAGDatabase(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model
    )

    stats = db.get_stats()
    print("RAG Database Statistics")
    print("=" * 40)
    for key, value in stats.items():
        print(f"  {key}: {value}")


def cmd_clear(args):
    """Clear the database."""
    db = RAGDatabase(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model
    )

    if not args.yes:
        confirm = input(f"Clear collection '{args.collection}'? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    db.clear()
    print("Database cleared.")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Tool - Create and query document databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create database:        python rag_tool.py create --input ./docs --name myproject
  Create with summaries:  python rag_tool.py create --input ./code --name mycode --summarize
  Query database:         python rag_tool.py query "What is the main function?" --name myproject
  Interactive mode:       python rag_tool.py interactive --name myproject
  Show stats:             python rag_tool.py stats --name myproject
"""
    )

    # Global options
    parser.add_argument(
        '--name', '-n',
        help='Database name (stores at ~/.rag_tool/<name>/)'
    )
    parser.add_argument(
        '--db-path',
        default=None,
        help=f'Database storage path (default: {DEFAULT_DB_PATH})'
    )
    parser.add_argument(
        '--collection',
        default=DEFAULT_COLLECTION_NAME,
        help=f'Collection name (default: {DEFAULT_COLLECTION_NAME})'
    )
    parser.add_argument(
        '--embedding-model',
        default=DEFAULT_EMBEDDING_MODEL,
        help=f'Ollama embedding model (default: {DEFAULT_EMBEDDING_MODEL})'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create RAG database from documents')
    create_parser.add_argument(
        '--name',
        dest='name_override',
        help='Database name (stores at ~/.rag_tool/<name>/)'
    )
    create_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing documents'
    )
    create_parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Size of document chunks (default: 1000)'
    )
    create_parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Overlap between chunks (default: 200)'
    )
    create_parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )
    create_parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing database instead of clearing'
    )
    create_parser.add_argument(
        '--summarize', '-s',
        action='store_true',
        help='Generate documentation summaries for code files using LLM'
    )
    create_parser.add_argument(
        '--summarize-model',
        default=DEFAULT_SUMMARIZER_MODEL,
        help=f'LLM model for code summarization (default: {DEFAULT_SUMMARIZER_MODEL})'
    )
    create_parser.set_defaults(func=cmd_create)

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG database')
    query_parser.add_argument(
        '--name',
        dest='name_override',
        help='Database name (stores at ~/.rag_tool/<name>/)'
    )
    query_parser.add_argument(
        'question',
        help='Question to ask'
    )
    query_parser.add_argument(
        '--results', '-r',
        type=int,
        default=5,
        help='Number of results to retrieve (default: 5)'
    )
    query_parser.add_argument(
        '--llm-model',
        default=DEFAULT_LLM_MODEL,
        help=f'Ollama LLM model for response generation (default: {DEFAULT_LLM_MODEL})'
    )
    query_parser.add_argument(
        '--search-only',
        action='store_true',
        help='Show search results without LLM generation'
    )
    query_parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Do not show source documents'
    )
    query_parser.set_defaults(func=cmd_query)

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive session')
    interactive_parser.add_argument(
        '--name',
        dest='name_override',
        help='Database name (stores at ~/.rag_tool/<name>/)'
    )
    interactive_parser.add_argument(
        '--llm-model',
        default=DEFAULT_LLM_MODEL,
        help=f'Ollama LLM model (default: {DEFAULT_LLM_MODEL})'
    )
    interactive_parser.set_defaults(func=cmd_interactive)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument(
        '--name',
        dest='name_override',
        help='Database name (stores at ~/.rag_tool/<name>/)'
    )
    stats_parser.set_defaults(func=cmd_stats)

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the database')
    clear_parser.add_argument(
        '--name',
        dest='name_override',
        help='Database name (stores at ~/.rag_tool/<name>/)'
    )
    clear_parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation'
    )
    clear_parser.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Resolve --name to --db-path (check both global and subcommand)
    name = getattr(args, 'name_override', None) or args.name
    if name:
        args.db_path = os.path.expanduser(f"~/.rag_tool/{name}")
    elif args.db_path is None:
        args.db_path = DEFAULT_DB_PATH

    args.func(args)


if __name__ == "__main__":
    main()
