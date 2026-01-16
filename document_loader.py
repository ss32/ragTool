"""
Document Loader Module
Handles loading and processing various document types for RAG ingestion.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# File extension mappings
TEXT_EXTENSIONS = {'.txt', '.text'}
MARKDOWN_EXTENSIONS = {'.md', '.markdown', '.rst'}
PDF_EXTENSIONS = {'.pdf'}
SOURCE_CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.cpp', '.c', '.h', '.hpp', '.cc',
    '.java', '.go', '.rs', '.rb',
    '.sh', '.bash', '.zsh',
    '.json', '.yaml', '.yml', '.toml',
    '.xml', '.html', '.css', '.scss',
    '.sql', '.r', '.scala', '.kt'
}

ALL_SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS | MARKDOWN_EXTENSIONS |
    PDF_EXTENSIONS | SOURCE_CODE_EXTENSIONS
)


def get_file_type(file_path: Path) -> str:
    """Determine the type of file based on extension."""
    ext = file_path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return 'text'
    elif ext in MARKDOWN_EXTENSIONS:
        return 'markdown'
    elif ext in PDF_EXTENSIONS:
        return 'pdf'
    elif ext in SOURCE_CODE_EXTENSIONS:
        return 'source_code'
    return 'unknown'


def load_text_file(file_path: Path) -> str:
    """Load a plain text or markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def load_pdf_file(file_path: Path) -> str:
    """Load a PDF file and extract text."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return '\n\n'.join(text_parts)
    except Exception as e:
        print(f"Warning: Could not load PDF {file_path}: {e}")
        return ""


def load_source_code(file_path: Path) -> str:
    """Load source code file with language context."""
    content = load_text_file(file_path)
    ext = file_path.suffix.lower()

    # Add language context as a header
    lang_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.cpp': 'C++',
        '.c': 'C',
        '.h': 'C/C++ Header',
        '.hpp': 'C++ Header',
        '.java': 'Java',
        '.go': 'Go',
        '.rs': 'Rust',
        '.rb': 'Ruby',
    }

    language = lang_map.get(ext, ext[1:].upper())
    return f"[{language} Source Code]\n\n{content}"


def discover_files(directory: str, recursive: bool = True) -> List[Path]:
    """Discover all supported files in a directory."""
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

    return sorted(files)


def load_document(file_path: Path) -> Optional[Document]:
    """Load a single document and return a LangChain Document object."""
    file_type = get_file_type(file_path)

    if file_type == 'unknown':
        return None

    try:
        if file_type == 'pdf':
            content = load_pdf_file(file_path)
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
    skip_files: Optional[set] = None
) -> List[Document]:
    """Load all documents from a directory.

    Args:
        directory: Path to the directory containing documents
        recursive: Whether to search subdirectories
        skip_files: Optional set of file paths to skip (for resume functionality)

    Returns:
        List of loaded Document objects
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

    documents = []
    for file_path in files:
        doc = load_document(file_path)
        if doc:
            documents.append(doc)
            print(f"  Loaded: {file_path.name}")

    print(f"Successfully loaded {len(documents)} documents")
    return documents


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
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    quiet: bool = False
) -> List[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of documents to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        quiet: If True, suppress progress output

    Returns:
        List of chunked Document objects
    """
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

    docs = load_documents(test_dir)
    chunks = chunk_documents(docs)

    print(f"\nSample chunk from first document:")
    if chunks:
        print(f"Content preview: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
