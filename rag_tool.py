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
import asyncio
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from document_loader import (
    load_documents, load_document, chunk_documents, get_file_list,
    load_with_docling, get_file_type, categorize_files, DOCLING_EXTENSIONS
)
from rag_database import (
    create_database, RAGDatabase,
    DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
)
from query_engine import QueryEngine, DEFAULT_LLM_MODEL
from code_summarizer import summarize_code_documents, CodeSummarizer, DEFAULT_SUMMARIZER_MODEL
from ingestion_tracker import IngestionTracker
from rag_gui import launch_gui


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
        "max_tokens": args.max_tokens,
        "summarize": args.summarize,
        "embedding_model": args.embedding_model
    }
    tracker.start_ingestion(input_dir, config)

    # Initialize database (don't clear if resuming)
    clear_existing = not resuming and not args.append
    db = RAGDatabase(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        hnsw_search_ef=args.hnsw_ef
    )

    if clear_existing:
        db.clear()

    print(f"\nDatabase path: {args.db_path}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {args.embedding_model}")

    # Initialize summarizer if needed
    summarizer = None
    summarize_workers = getattr(args, 'summarize_workers', 1)
    if args.summarize:
        print(f"Code summarization enabled (model: {args.summarize_model}, workers: {summarize_workers})")
        summarizer = CodeSummarizer(model=args.summarize_model)

    # Categorize files for different processing paths
    docling_files, text_files, source_code_files = categorize_files(files_to_process)
    non_docling_files = text_files + source_code_files

    print(f"\nFile types:")
    print(f"  Docling documents (PDF, DOCX, etc.): {len(docling_files)}")
    print(f"  Text/markdown files: {len(text_files)}")
    print(f"  Source code files: {len(source_code_files)}")

    # Process files incrementally
    print("\nProcessing files...")
    processed_count = 0
    total_chunks = 0

    # Process docling files (batch processing is more efficient)
    if docling_files:
        print(f"\nProcessing {len(docling_files)} documents with docling...")
        try:
            docling_chunks = load_with_docling(docling_files, max_tokens=args.max_tokens)
            if docling_chunks:
                db.add_documents(docling_chunks, quiet=True)
                total_chunks += len(docling_chunks)
                print(f"  Added {len(docling_chunks)} chunks from docling")

            # Mark all docling files as processed
            for file_path in docling_files:
                tracker.mark_file_ingested(str(file_path))
                processed_count += 1
                print(f"  Processed: {file_path.name}")

        except KeyboardInterrupt:
            print(f"\n\nInterrupted during docling processing! Progress saved.")
            print(f"  Run the same command to resume.")
            sys.exit(0)
        except Exception as e:
            print(f"  Error in docling batch processing: {e}")
            print(f"  Falling back to individual file processing...")
            # Fall back to processing docling files individually
            non_docling_files = docling_files + non_docling_files

    # Process non-docling files (text, markdown, source code)
    # Use batch accumulation for more efficient embedding generation
    batch_size = getattr(args, 'batch_size', 100)
    workers = getattr(args, 'workers', 1)
    accumulated_chunks = []
    accumulated_files = []

    def flush_batch(force_save: bool = False):
        """Add accumulated chunks to database and mark files as processed."""
        nonlocal total_chunks, processed_count, accumulated_chunks, accumulated_files
        if not accumulated_chunks:
            return
        db.add_documents(accumulated_chunks, quiet=True)
        total_chunks += len(accumulated_chunks)
        # Batch update tracker (more efficient than individual saves)
        tracker.mark_files_ingested([str(f) for f in accumulated_files], force_save=force_save)
        processed_count += len(accumulated_files)
        print(f"  Batch added: {len(accumulated_chunks)} chunks from {len(accumulated_files)} files")
        accumulated_chunks = []
        accumulated_files = []

    def load_and_chunk_file(file_path):
        """Load a single file and return its chunks (for parallel processing)."""
        try:
            doc = load_document(file_path)
            if not doc:
                return file_path, [], None

            file_type = get_file_type(file_path)
            if file_type == 'docling':
                chunks = [doc]
            else:
                chunks = chunk_documents([doc], args.chunk_size, args.chunk_overlap, quiet=True)

            return file_path, chunks, doc
        except Exception as e:
            return file_path, [], str(e)

    if workers > 1 and not summarizer:
        # Parallel file loading (only when not summarizing - summarization needs sequential LLM calls)
        print(f"\nUsing {workers} workers for parallel file loading...")
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(load_and_chunk_file, f): f for f in non_docling_files}
                completed = 0

                for future in as_completed(futures):
                    file_path, chunks, error = future.result()
                    completed += 1

                    if error and isinstance(error, str):
                        print(f"[{completed}/{len(non_docling_files)}] {file_path.name} - Error: {error}")
                        continue

                    if not chunks:
                        continue

                    print(f"[{completed}/{len(non_docling_files)}] {file_path.name} - {len(chunks)} chunks")

                    accumulated_chunks.extend(chunks)
                    accumulated_files.append(file_path)

                    if len(accumulated_chunks) >= batch_size:
                        flush_batch()

        except KeyboardInterrupt:
            if accumulated_chunks:
                print(f"\n\nSaving pending batch before exit...")
                flush_batch(force_save=True)
            tracker.flush()
            print(f"\n\nInterrupted! Progress saved.")
            print(f"  Processed {processed_count}/{len(files_to_process)} files")
            print(f"  Run the same command to resume.")
            sys.exit(0)
    elif summarizer and summarize_workers > 1:
        # Parallel summarization mode: Load all files, batch summarize, then chunk
        print(f"\nUsing parallel summarization with {summarize_workers} workers...")

        # First, load all documents
        print("Loading documents...")
        loaded_docs = []
        for i, file_path in enumerate(non_docling_files):
            try:
                doc = load_document(file_path)
                if doc:
                    loaded_docs.append((file_path, doc))
                    print(f"  [{i + 1}/{len(non_docling_files)}] Loaded: {file_path.name}")
            except Exception as e:
                print(f"  [{i + 1}/{len(non_docling_files)}] Error loading {file_path.name}: {e}")

        # Batch summarize code files in parallel
        code_docs = [(fp, doc) for fp, doc in loaded_docs if doc.metadata.get('file_type') == 'source_code']
        other_docs = [(fp, doc) for fp, doc in loaded_docs if doc.metadata.get('file_type') != 'source_code']

        if code_docs:
            print(f"\nSummarizing {len(code_docs)} code files with {summarize_workers} workers...")
            docs_to_summarize = [doc for _, doc in code_docs]
            summarized_docs = summarizer.summarize_documents_parallel(
                docs_to_summarize,
                max_workers=summarize_workers,
                show_progress=True
            )
            # Rebuild code_docs with summarized versions
            code_docs = [(code_docs[i][0], summarized_docs[i]) for i in range(len(code_docs))]

        # Combine and process all documents
        all_docs = code_docs + other_docs
        print(f"\nChunking and adding {len(all_docs)} documents...")

        try:
            for file_path, doc in all_docs:
                file_type = get_file_type(file_path)
                if file_type == 'docling':
                    chunks = [doc]
                else:
                    chunks = chunk_documents([doc], args.chunk_size, args.chunk_overlap, quiet=True)

                if chunks:
                    accumulated_chunks.extend(chunks)
                    accumulated_files.append(file_path)

                if len(accumulated_chunks) >= batch_size:
                    flush_batch()

        except KeyboardInterrupt:
            if accumulated_chunks:
                print(f"\n\nSaving pending batch before exit...")
                flush_batch(force_save=True)
            tracker.flush()
            print(f"\n\nInterrupted! Progress saved.")
            print(f"  Processed {processed_count}/{len(files_to_process)} files")
            print(f"  Run the same command to resume.")
            sys.exit(0)

    else:
        # Sequential processing (default)
        for i, file_path in enumerate(non_docling_files):
            try:
                # Load single document
                doc = load_document(file_path)
                if not doc:
                    continue

                print(f"[{i + 1}/{len(non_docling_files)}] {file_path.name}")

                # Summarize if it's code and summarization is enabled
                if summarizer and doc.metadata.get('file_type') == 'source_code':
                    print(f"  Summarizing...")
                    doc = summarizer.summarize_document(doc)

                # Chunk the document (docling files are already chunked)
                file_type = get_file_type(file_path)
                if file_type == 'docling':
                    # Single docling file fallback - already chunked by load_document
                    chunks = [doc]
                else:
                    chunks = chunk_documents([doc], args.chunk_size, args.chunk_overlap, quiet=True)

                # Accumulate chunks for batch processing
                if chunks:
                    accumulated_chunks.extend(chunks)
                    accumulated_files.append(file_path)
                    print(f"  Queued {len(chunks)} chunks (batch: {len(accumulated_chunks)})")

                # Flush batch when threshold reached
                if len(accumulated_chunks) >= batch_size:
                    flush_batch()

            except KeyboardInterrupt:
                # Save any pending work before exiting
                if accumulated_chunks:
                    print(f"\n\nSaving pending batch before exit...")
                    flush_batch(force_save=True)
                tracker.flush()  # Ensure any pending tracker updates are saved
                print(f"\n\nInterrupted! Progress saved.")
                print(f"  Processed {processed_count}/{len(files_to_process)} files")
                print(f"  Run the same command to resume.")
                sys.exit(0)
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue

    # Flush any remaining chunks
    if accumulated_chunks:
        flush_batch(force_save=True)

    # Ensure all tracker data is persisted
    tracker.flush()

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
    use_async = getattr(args, 'use_async', False)

    if use_async:
        asyncio.run(_cmd_query_async(args))
    else:
        _cmd_query_sync(args)


def _cmd_query_sync(args):
    """Synchronous query implementation."""
    engine = QueryEngine(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        warm_up=not args.no_warmup,
        enable_response_cache=not args.no_cache,
        hnsw_search_ef=args.hnsw_ef
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
    elif args.stream:
        for token in engine.query_stream(
            question,
            n_results=args.results,
            show_sources=not args.no_sources
        ):
            print(token, end='', flush=True)
        print()
    else:
        answer = engine.query(
            question,
            n_results=args.results,
            show_sources=not args.no_sources
        )
        print(answer)


async def _cmd_query_async(args):
    """Async query implementation."""
    engine = QueryEngine(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        warm_up=not args.no_warmup,
        enable_response_cache=not args.no_cache,
        hnsw_search_ef=args.hnsw_ef
    )

    stats = engine.db.get_stats()
    if stats['document_count'] == 0:
        print("Error: Database is empty. Run 'create' first.")
        sys.exit(1)

    question = args.question
    print(f"Question: {question} (async mode)")
    print("-" * 60)

    if args.search_only:
        results = await engine.search_only_async(question, n_results=args.results)
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n[{i + 1}] Distance: {result['distance']:.4f}")
            print(f"    Source: {result['metadata'].get('source', 'unknown')}")
            print(f"    Content: {result['content'][:300]}...")
    elif args.stream:
        async for token in engine.query_stream_async(
            question,
            n_results=args.results,
            show_sources=not args.no_sources
        ):
            print(token, end='', flush=True)
        print()
    else:
        answer = await engine.query_async(
            question,
            n_results=args.results,
            show_sources=not args.no_sources
        )
        print(answer)


def cmd_interactive(args):
    """Start interactive query session."""
    use_async = getattr(args, 'use_async', False)

    engine = QueryEngine(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        warm_up=not args.no_warmup,
        enable_response_cache=not args.no_cache,
        hnsw_search_ef=args.hnsw_ef
    )

    stats = engine.db.get_stats()
    if stats['document_count'] == 0:
        print("Error: Database is empty. Run 'create' first.")
        sys.exit(1)

    if use_async:
        asyncio.run(engine.interactive_session_async(use_streaming=not args.no_stream))
    else:
        engine.interactive_session(use_streaming=not args.no_stream)


def cmd_stats(args):
    """Show database statistics."""
    db = RAGDatabase(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        hnsw_search_ef=args.hnsw_ef
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
        embedding_model=args.embedding_model,
        hnsw_search_ef=args.hnsw_ef
    )

    if not args.yes:
        confirm = input(f"Clear collection '{args.collection}'? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    db.clear()
    print("Database cleared.")


def cmd_clear_cache(args):
    """Clear embedding and response caches."""
    from pathlib import Path

    print("Clearing caches...")

    # Clear response cache file
    cache_file = Path(args.db_path) / "response_cache.json"
    if cache_file.exists():
        cache_file.unlink()
        print(f"  Deleted response cache: {cache_file}")
    else:
        print(f"  No response cache found at: {cache_file}")

    # Note: Embedding cache is in-memory and session-scoped, so it's cleared on restart
    print("  Embedding cache is session-scoped (cleared on restart)")
    print("\nCaches cleared successfully.")


def cmd_benchmark(args):
    """Benchmark query performance."""
    import time

    print("RAG Query Performance Benchmark")
    print("=" * 60)

    # Initialize engine with warm-up disabled for accurate measurement
    print("\nInitializing query engine...")
    init_start = time.perf_counter()
    engine = QueryEngine(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        warm_up=False,
        enable_response_cache=True,
        hnsw_search_ef=args.hnsw_ef
    )
    init_time = time.perf_counter() - init_start
    print(f"  Initialization time: {init_time:.2f}s")

    stats = engine.db.get_stats()
    if stats['document_count'] == 0:
        print("Error: Database is empty. Run 'create' first.")
        sys.exit(1)

    print(f"  Documents in database: {stats['document_count']}")

    # Warm up
    print("\nWarming up LLM model...")
    warmup_start = time.perf_counter()
    engine._warm_up_model()
    warmup_time = time.perf_counter() - warmup_start
    print(f"  Warm-up time: {warmup_time:.2f}s")

    # Test queries
    test_queries = args.queries if args.queries else [
        "What is the main purpose of this code?",
        "How does error handling work?",
        "What are the key functions?",
    ]

    print(f"\nRunning {len(test_queries)} test queries...")
    print("-" * 60)

    results = []
    for i, query in enumerate(test_queries):
        print(f"\n[{i + 1}] Query: {query[:50]}{'...' if len(query) > 50 else ''}")

        # Clear embedding cache for cold test
        engine.db.clear_embedding_cache()

        # Cold query (no cache)
        cold_start = time.perf_counter()
        engine.query(query, show_sources=False)
        cold_time = time.perf_counter() - cold_start
        print(f"    Cold (no cache): {cold_time:.2f}s")

        # Warm query (with embedding cache, response cache)
        warm_start = time.perf_counter()
        engine.query(query, show_sources=False)
        warm_time = time.perf_counter() - warm_start
        print(f"    Warm (cached):   {warm_time:.3f}s")

        # Search only (no LLM)
        search_start = time.perf_counter()
        engine.search_only(query)
        search_time = time.perf_counter() - search_start
        print(f"    Search only:     {search_time:.3f}s")

        results.append({
            'query': query,
            'cold': cold_time,
            'warm': warm_time,
            'search': search_time
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_cold = sum(r['cold'] for r in results) / len(results)
    avg_warm = sum(r['warm'] for r in results) / len(results)
    avg_search = sum(r['search'] for r in results) / len(results)

    print(f"\nAverage times:")
    print(f"  Cold query (no cache):     {avg_cold:.2f}s")
    print(f"  Warm query (cached):       {avg_warm:.3f}s")
    print(f"  Search only (no LLM):      {avg_search:.3f}s")

    cache_stats = engine.get_cache_stats()
    print(f"\nCache statistics:")
    print(f"  Embedding cache hit rate:  {cache_stats['hit_rate']}")
    print(f"  Response cache hit rate:   {cache_stats['response_cache_hit_rate']}")

    speedup = avg_cold / avg_warm if avg_warm > 0 else 0
    print(f"\nCache speedup: {speedup:.1f}x faster with caching")


def cmd_gui(args):
    """Launch the interactive GUI."""
    print("Starting RAG Tool GUI...")
    print(f"  LLM Model: {args.llm_model}")
    print(f"  Server: http://{args.host}:{args.port}")
    if args.share:
        print("  Creating public share URL...")
    print()

    launch_gui(
        llm_model=args.llm_model,
        share=args.share,
        server_port=args.port,
        server_name=args.host
    )


def main():
    parser = argparse.ArgumentParser(
        description="RAG Tool - Create and query document databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create database:        python rag_tool.py create --input ./docs --name myproject
  Create with summaries:  python rag_tool.py create --input ./code --name mycode --summarize
  Query database:         python rag_tool.py query "What is the main function?" --name myproject
  Query with streaming:   python rag_tool.py query "What does this do?" --name myproject --stream
  Interactive mode:       python rag_tool.py interactive --name myproject
  Web GUI:                python rag_tool.py gui
  Show stats:             python rag_tool.py stats --name myproject
  Clear caches:           python rag_tool.py clear-cache --name myproject
  Benchmark:              python rag_tool.py benchmark --name myproject
  Fast search (lower accuracy): python rag_tool.py query "question" --hnsw-ef 50
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
        help=f'Ollama embedding model (default: {DEFAULT_EMBEDDING_MODEL}). '
             f'Options: all-minilm:22m (fastest), all-minilm:33m (balanced), '
             f'nomic-embed-text (best quality)'
    )
    parser.add_argument(
        '--hnsw-ef',
        type=int,
        default=100,
        help='HNSW search_ef parameter for ChromaDB vector search (default: 100). '
             'Trade-off: Lower values (e.g., 50) = faster search but slightly less accurate. '
             'Higher values (e.g., 200) = more accurate but slower. '
             'Range 50-200 recommended.'
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
        default=400,
        help='Size of document chunks for text/code files in characters (default: 400). '
             'Keep under ~400 chars to stay within the 512 token limit of embedding models.'
    )
    create_parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=80,
        help='Overlap between chunks for text/code files (default: 80)'
    )
    create_parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Max tokens per chunk for docling HybridChunker (default: 512)'
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
    create_parser.add_argument(
        '--summarize-workers',
        type=int,
        default=1,
        help='Number of parallel workers for code summarization (default: 1). '
             'Higher values speed up summarization but use more GPU memory. '
             'Recommended: 2-4 depending on your GPU.'
    )
    create_parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of chunks to accumulate before batch embedding (default: 100). '
             'Higher values = fewer embedding calls but more memory usage.'
    )
    create_parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for file loading (default: 1). '
             'Higher values speed up I/O-bound loading. '
             'Note: Parallel loading is disabled when --summarize is used.'
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
    query_parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream response tokens as they are generated'
    )
    query_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable response caching'
    )
    query_parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip LLM model warm-up (faster startup, slower first query)'
    )
    query_parser.add_argument(
        '--async',
        dest='use_async',
        action='store_true',
        help='Use async operations for non-blocking I/O. '
             'Enables concurrent embedding and search operations.'
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
    interactive_parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Disable response streaming (wait for full response)'
    )
    interactive_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable response caching'
    )
    interactive_parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip LLM model warm-up (faster startup, slower first query)'
    )
    interactive_parser.add_argument(
        '--async',
        dest='use_async',
        action='store_true',
        help='Use async operations for non-blocking I/O. '
             'Enables concurrent embedding and search operations.'
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

    # Clear-cache command
    clear_cache_parser = subparsers.add_parser(
        'clear-cache',
        help='Clear embedding and response caches'
    )
    clear_cache_parser.add_argument(
        '--name',
        dest='name_override',
        help='Database name (stores at ~/.rag_tool/<name>/)'
    )
    clear_cache_parser.set_defaults(func=cmd_clear_cache)

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark query performance')
    benchmark_parser.add_argument(
        '--name',
        dest='name_override',
        help='Database name (stores at ~/.rag_tool/<name>/)'
    )
    benchmark_parser.add_argument(
        '--llm-model',
        default=DEFAULT_LLM_MODEL,
        help=f'Ollama LLM model (default: {DEFAULT_LLM_MODEL})'
    )
    benchmark_parser.add_argument(
        '--queries', '-q',
        nargs='+',
        help='Custom queries to benchmark (default: built-in test queries)'
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch interactive web GUI')
    gui_parser.add_argument(
        '--llm-model',
        default=DEFAULT_LLM_MODEL,
        help=f'Ollama LLM model (default: {DEFAULT_LLM_MODEL})'
    )
    gui_parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Server port (default: 7860)'
    )
    gui_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Server host (default: 127.0.0.1, use 0.0.0.0 for all interfaces)'
    )
    gui_parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public URL for sharing'
    )
    gui_parser.set_defaults(func=cmd_gui)

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
