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

from document_loader import load_documents, load_document, chunk_documents, get_file_list
from rag_database import (
    create_database, RAGDatabase,
    DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
)
from query_engine import QueryEngine, DEFAULT_LLM_MODEL
from code_summarizer import summarize_code_documents, CodeSummarizer, DEFAULT_SUMMARIZER_MODEL
from ingestion_tracker import IngestionTracker


def cmd_create(args):
    """Create a new RAG database from documents."""
    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    # Initialize tracker for resume functionality
    tracker = IngestionTracker(args.db_path)

    # Check for existing progress
    resuming = False
    skip_files = set()

    if tracker.has_progress() and not args.fresh:
        prev_input = tracker.get_input_directory()
        if prev_input == input_dir:
            stats = tracker.get_stats()
            print(f"Resuming previous ingestion...")
            print(f"  Previously processed: {stats['ingested_count']} files")
            print(f"  Started: {stats['started_at']}")
            skip_files = tracker.get_ingested_files()
            resuming = True
        else:
            print(f"Warning: Previous ingestion was from a different directory:")
            print(f"  Previous: {prev_input}")
            print(f"  Current:  {input_dir}")
            print(f"Use --fresh to start over, or use the same input directory to resume.")
            sys.exit(1)
    elif args.fresh and tracker.has_progress():
        print("Starting fresh (clearing previous progress)...")
        tracker.clear()

    print(f"Creating RAG database from: {input_dir}")
    print("=" * 60)

    # Get list of all files to process
    all_files = get_file_list(input_dir, recursive=not args.no_recursive)

    if not all_files:
        print("No supported documents found in the directory.")
        sys.exit(1)

    # Filter out already-processed files
    files_to_process = [f for f in all_files if str(f) not in skip_files]

    if not files_to_process:
        print("All files have already been processed!")
        print("Use --fresh to re-process all files.")
        sys.exit(0)

    print(f"Total files: {len(all_files)}")
    if skip_files:
        print(f"Already processed: {len(skip_files)}")
    print(f"Files to process: {len(files_to_process)}")

    # Record ingestion config
    config = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "summarize": args.summarize,
        "embedding_model": args.embedding_model
    }
    tracker.start_ingestion(input_dir, config)

    # Initialize database (don't clear if resuming)
    clear_existing = not resuming and not args.append
    db = RAGDatabase(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model
    )

    if clear_existing:
        db.clear()

    print(f"\nDatabase path: {args.db_path}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {args.embedding_model}")

    # Initialize summarizer if needed
    summarizer = None
    if args.summarize:
        print(f"Code summarization enabled (model: {args.summarize_model})")
        summarizer = CodeSummarizer(model=args.summarize_model)

    # Process files incrementally
    print("\nProcessing files...")
    processed_count = 0
    total_chunks = 0

    for i, file_path in enumerate(files_to_process):
        try:
            # Load single document
            doc = load_document(file_path)
            if not doc:
                continue

            print(f"[{i + 1}/{len(files_to_process)}] {file_path.name}")

            # Summarize if it's code and summarization is enabled
            if summarizer and doc.metadata.get('file_type') == 'source_code':
                print(f"  Summarizing...")
                doc = summarizer.summarize_document(doc)

            # Chunk the document
            chunks = chunk_documents([doc], args.chunk_size, args.chunk_overlap, quiet=True)

            # Add to database
            if chunks:
                db.add_documents(chunks, quiet=True)
                total_chunks += len(chunks)
                print(f"  Added {len(chunks)} chunks")

            # Mark as processed
            tracker.mark_file_ingested(str(file_path))
            processed_count += 1

        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Progress saved.")
            print(f"  Processed {processed_count}/{len(files_to_process)} files")
            print(f"  Run the same command to resume.")
            sys.exit(0)
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Database creation complete!")
    stats = db.get_stats()
    print(f"  Files processed this run: {processed_count}")
    print(f"  Chunks added this run: {total_chunks}")
    print(f"  Total chunks in database: {stats['document_count']}")
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
        '--fresh',
        action='store_true',
        help='Start fresh, ignoring any previous progress (clears existing data)'
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
