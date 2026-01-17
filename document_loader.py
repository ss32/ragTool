"""
Document Loader Module
Handles loading and processing various document types for RAG ingestion.

Uses docling for document processing (PDFs, Office docs, HTML, images)
and preserves existing source code handling with language context headers.
"""

import os
import warnings

# Suppress the "Token indices sequence length" warning from transformers tokenizers
# This warning occurs when docling's HybridChunker tokenizes documents longer than
# the tokenizer's max length before chunking. The chunking still works correctly.
warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum sequence length"
)
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# File type categorization
# Docling handles these document types with advanced processing
DOCLING_EXTENSIONS = {
    '.pdf', '.docx', '.pptx', '.xlsx',
    '.html', '.htm',
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp'
}

# Text/markdown can use docling for better structure handling
TEXT_EXTENSIONS = {'.txt', '.text', '.md', '.markdown', '.rst'}

# Source code keeps existing handling with language context headers
SOURCE_CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.cpp', '.c', '.h', '.hpp', '.cc',
    '.java', '.go', '.rs', '.rb',
    '.sh', '.bash', '.zsh',
    '.json', '.yaml', '.yml', '.toml',
    '.xml', '.css', '.scss',
    '.sql', '.r', '.scala', '.kt'
}

ALL_SUPPORTED_EXTENSIONS = (
    DOCLING_EXTENSIONS | TEXT_EXTENSIONS | SOURCE_CODE_EXTENSIONS
)

# Minimum file size in bytes to process (skip empty/tiny files)
MIN_FILE_SIZE = 10

# Language context mapping for source code
LANG_MAP = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.ts': 'TypeScript',
    '.jsx': 'JavaScript (JSX)',
    '.tsx': 'TypeScript (TSX)',
    '.cpp': 'C++',
    '.c': 'C',
    '.h': 'C/C++ Header',
    '.hpp': 'C++ Header',
    '.cc': 'C++',
    '.java': 'Java',
    '.go': 'Go',
    '.rs': 'Rust',
    '.rb': 'Ruby',
    '.sh': 'Shell',
    '.bash': 'Bash',
    '.zsh': 'Zsh',
    '.json': 'JSON',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.toml': 'TOML',
    '.xml': 'XML',
    '.css': 'CSS',
    '.scss': 'SCSS',
    '.sql': 'SQL',
    '.r': 'R',
    '.scala': 'Scala',
    '.kt': 'Kotlin',
}


def get_file_type(file_path: Path) -> str:
    """Determine the type of file based on extension."""
    ext = file_path.suffix.lower()
    if ext in DOCLING_EXTENSIONS:
        return 'docling'
    elif ext in TEXT_EXTENSIONS:
        return 'text'
    elif ext in SOURCE_CODE_EXTENSIONS:
        return 'source_code'
    return 'unknown'


def load_text_file(file_path: Path) -> str:
    """Load a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def load_source_code(file_path: Path) -> str:
    """Load source code file with language context header."""
    content = load_text_file(file_path)
    ext = file_path.suffix.lower()
    language = LANG_MAP.get(ext, ext[1:].upper())
    return f"[{language} Source Code]\n\n{content}"


def load_with_docling(
    file_paths: List[Path],
    max_tokens: int = 512,
    quiet: bool = False
) -> List[Document]:
    """Load documents using DoclingLoader with HybridChunker.

    Args:
        file_paths: List of file paths to load
        max_tokens: Maximum tokens per chunk for HybridChunker
        quiet: If True, suppress progress output

    Returns:
        List of chunked Document objects
    """
    try:
        from langchain_docling import DoclingLoader
        from langchain_docling.loader import ExportType
        from docling.chunking import HybridChunker
    except ImportError as e:
        print(f"Warning: docling not available ({e}). Falling back to basic text loading.")
        return []

    if not file_paths:
        return []

    # Convert paths to strings for DoclingLoader
    file_path_strs = [str(p) for p in file_paths]

    try:
        # Create HybridChunker for intelligent chunking
        chunker = HybridChunker(max_tokens=max_tokens)

        # Create loader with chunking enabled
        loader = DoclingLoader(
            file_path=file_path_strs,
            export_type=ExportType.DOC_CHUNKS,
            chunker=chunker,
        )

        docs = loader.load()

        if not quiet:
            print(f"  Docling processed {len(file_paths)} files into {len(docs)} chunks")

        return docs

    except Exception as e:
        print(f"Warning: Docling processing failed: {e}")
        return []


def discover_files(
    directory: str,
    recursive: bool = True,
    min_size: int = MIN_FILE_SIZE
) -> List[Path]:
    """Discover all supported files in a directory.

    Args:
        directory: Path to the directory to search
        recursive: Whether to search subdirectories
        min_size: Minimum file size in bytes (default: MIN_FILE_SIZE). Files smaller
                  than this are excluded to avoid processing empty/tiny files.

    Returns:
        Sorted list of Path objects for supported files
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    files = []
    if recursive:
        for ext in ALL_SUPPORTED_EXTENSIONS:
            files.extend(dir_path.rglob(f'*{ext}'))
    else:
        for ext in ALL_SUPPORTED_EXTENSIONS:
            files.extend(dir_path.glob(f'*{ext}'))

    # Filter out hidden files and directories
    files = [f for f in files if not any(part.startswith('.') for part in f.parts)]

    # Filter out empty/tiny files
    if min_size > 0:
        files = [f for f in files if f.stat().st_size >= min_size]

    return sorted(files)


def categorize_files(files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """Categorize files into docling, text, and source code lists.

    Returns:
        Tuple of (docling_files, text_files, source_code_files)
    """
    docling_files = []
    text_files = []
    source_code_files = []

    for f in files:
        file_type = get_file_type(f)
        if file_type == 'docling':
            docling_files.append(f)
        elif file_type == 'text':
            text_files.append(f)
        elif file_type == 'source_code':
            source_code_files.append(f)

    return docling_files, text_files, source_code_files


def load_document(file_path: Path, min_size: int = MIN_FILE_SIZE) -> Optional[Document]:
    """Load a single document and return a LangChain Document object.

    Note: For docling-supported files, prefer using load_with_docling for batch processing.
    This function is kept for single-file loading and source code files.

    Args:
        file_path: Path to the file to load
        min_size: Minimum file size in bytes (default: MIN_FILE_SIZE). Files smaller
                  than this are skipped to avoid processing empty/tiny files.

    Returns:
        Document object or None if file is too small, unknown type, or cannot be loaded.
    """
    file_type = get_file_type(file_path)

    if file_type == 'unknown':
        return None

    # Quick size check to skip empty/tiny files
    try:
        if file_path.stat().st_size < min_size:
            return None
    except OSError:
        return None

    try:
        if file_type == 'docling':
            # For single docling file, use docling loader
            docs = load_with_docling([file_path], quiet=True)
            if docs:
                # Return first chunk with combined metadata
                return docs[0]
            # Fallback to text loading for docling-supported files
            content = load_text_file(file_path)
        elif file_type == 'source_code':
            content = load_source_code(file_path)
        else:
            content = load_text_file(file_path)

        if not content.strip():
            return None

        metadata = {
            'source': str(file_path),
            'file_name': file_path.name,
            'file_type': file_type,
            'extension': file_path.suffix.lower()
        }

        return Document(page_content=content, metadata=metadata)

    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None


def load_documents(
    directory: str,
    recursive: bool = True,
    skip_files: Optional[set] = None,
    max_tokens: int = 512
) -> Tuple[List[Document], List[Document]]:
    """Load all documents from a directory.

    Args:
        directory: Path to the directory containing documents
        recursive: Whether to search subdirectories
        skip_files: Optional set of file paths to skip (for resume functionality)
        max_tokens: Maximum tokens per chunk for docling HybridChunker

    Returns:
        Tuple of (docling_docs, other_docs) where:
        - docling_docs: Already chunked by docling's HybridChunker
        - other_docs: Need to be chunked separately (source code, plain text)
    """
    files = discover_files(directory, recursive)

    # Filter out already-processed files if resuming
    if skip_files:
        original_count = len(files)
        files = [f for f in files if str(f) not in skip_files]
        skipped = original_count - len(files)
        if skipped > 0:
            print(f"Found {original_count} supported files, skipping {skipped} already processed")
        else:
            print(f"Found {len(files)} supported files in {directory}")
    else:
        print(f"Found {len(files)} supported files in {directory}")

    # Categorize files
    docling_files, text_files, source_code_files = categorize_files(files)

    print(f"  Docling documents: {len(docling_files)}")
    print(f"  Text/markdown files: {len(text_files)}")
    print(f"  Source code files: {len(source_code_files)}")

    docling_docs = []
    other_docs = []

    # Process docling-supported files (batch processing)
    if docling_files:
        print("\nProcessing documents with docling...")
        docling_docs = load_with_docling(docling_files, max_tokens=max_tokens)
        for f in docling_files:
            print(f"  Loaded: {f.name}")

    # Process text/markdown files
    # These can also use docling for better structure, but keeping separate for flexibility
    for file_path in text_files:
        doc = Document(
            page_content=load_text_file(file_path),
            metadata={
                'source': str(file_path),
                'file_name': file_path.name,
                'file_type': 'text',
                'extension': file_path.suffix.lower()
            }
        )
        if doc.page_content.strip():
            other_docs.append(doc)
            print(f"  Loaded: {file_path.name}")

    # Process source code files (preserving language context headers)
    for file_path in source_code_files:
        content = load_source_code(file_path)
        if content.strip():
            doc = Document(
                page_content=content,
                metadata={
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': 'source_code',
                    'extension': file_path.suffix.lower()
                }
            )
            other_docs.append(doc)
            print(f"  Loaded: {file_path.name}")

    total_loaded = len(docling_docs) + len(other_docs)
    print(f"\nSuccessfully loaded: {len(docling_docs)} docling chunks + {len(other_docs)} other documents")

    return docling_docs, other_docs


def get_file_list(directory: str, recursive: bool = True) -> List[Path]:
    """Get list of supported files without loading them.

    Args:
        directory: Path to the directory containing documents
        recursive: Whether to search subdirectories

    Returns:
        List of Path objects for supported files
    """
    return discover_files(directory, recursive)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    quiet: bool = False
) -> List[Document]:
    """Split documents into smaller chunks for embedding.

    This is used for source code and text files. Docling documents are
    already chunked by HybridChunker and should not be passed here.

    Args:
        documents: List of documents to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        quiet: If True, suppress progress output

    Returns:
        List of chunked Document objects
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    if not quiet:
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    # Test the document loader
    import sys
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        test_dir = "/tmp/inbounds"

    docling_docs, other_docs = load_documents(test_dir)
    other_chunks = chunk_documents(other_docs)

    all_chunks = docling_docs + other_chunks

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"  From docling: {len(docling_docs)}")
    print(f"  From text/code: {len(other_chunks)}")

    if all_chunks:
        print(f"\nSample chunk:")
        print(f"Content preview: {all_chunks[0].page_content[:200]}...")
        print(f"Metadata: {all_chunks[0].metadata}")
