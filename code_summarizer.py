"""
Code Summarizer Module
Uses LLM to generate documentation/summaries for code chunks.
"""

import sys
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from ollama_utils import ensure_model_available, OllamaModelError

DEFAULT_SUMMARIZER_MODEL = "qwen3:8b"


class CodeSummarizer:
    """Generates summaries and documentation for code chunks using LLM."""

    def __init__(self, model: str = DEFAULT_SUMMARIZER_MODEL):
        # Ensure model is available (download if needed)
        try:
            ensure_model_available(model)
        except OllamaModelError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        self.llm = ChatOllama(model=model, temperature=0.1)
        self.model = model

    def summarize_code(self, code: str, file_name: str, language: str) -> str:
        """Generate a summary/documentation for a code chunk."""
        prompt = f"""Analyze this {language} code from {file_name} and provide a concise summary.
Include:
- What this code does (main purpose)
- Key functions/classes/methods and their purposes
- Important variables or data structures
- Any notable patterns or algorithms used

Be concise but comprehensive. Focus on what would help someone understand this code.

CODE:
```{language.lower()}
{code}
```

SUMMARY:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Warning: Could not summarize code: {e}")
            return ""

    def summarize_document(self, doc: Document) -> Document:
        """Add a summary to a document containing code."""
        file_type = doc.metadata.get('file_type', '')
        if file_type != 'source_code':
            return doc

        file_name = doc.metadata.get('file_name', 'unknown')
        extension = doc.metadata.get('extension', '')

        # Map extensions to language names
        lang_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'JavaScript/React',
            '.tsx': 'TypeScript/React',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.hpp': 'C++ Header',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.cs': 'C#',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.php': 'PHP',
        }
        language = lang_map.get(extension, extension[1:].upper() if extension else 'Code')

        # Generate summary
        summary = self.summarize_code(doc.page_content, file_name, language)

        if summary:
            # Prepend summary to the code content
            enhanced_content = f"""[DOCUMENTATION]
{summary}

[SOURCE CODE - {file_name}]
{doc.page_content}"""

            # Create new document with enhanced content
            new_doc = Document(
                page_content=enhanced_content,
                metadata={**doc.metadata, 'has_summary': True}
            )
            return new_doc

        return doc

    def summarize_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> List[Document]:
        """Add summaries to all code documents (sequential)."""
        summarized = []
        code_docs = [d for d in documents if d.metadata.get('file_type') == 'source_code']
        other_docs = [d for d in documents if d.metadata.get('file_type') != 'source_code']

        if show_progress and code_docs:
            print(f"\nGenerating summaries for {len(code_docs)} code files...")

        for i, doc in enumerate(code_docs):
            if show_progress:
                file_name = doc.metadata.get('file_name', 'unknown')
                print(f"  [{i + 1}/{len(code_docs)}] Summarizing: {file_name}")

            summarized_doc = self.summarize_document(doc)
            summarized.append(summarized_doc)

        if show_progress and code_docs:
            print(f"Summarization complete.")

        # Return summarized code docs + unchanged other docs
        return summarized + other_docs

    def summarize_documents_parallel(
        self,
        documents: List[Document],
        max_workers: int = 3,
        show_progress: bool = True
    ) -> List[Document]:
        """Add summaries to all code documents using parallel LLM calls.

        Args:
            documents: List of documents to process
            max_workers: Number of parallel workers for LLM calls (default: 3)
            show_progress: Whether to show progress output

        Returns:
            List of documents with summaries added to code files
        """
        code_docs = [d for d in documents if d.metadata.get('file_type') == 'source_code']
        other_docs = [d for d in documents if d.metadata.get('file_type') != 'source_code']

        if not code_docs:
            return documents

        if show_progress:
            print(f"\nGenerating summaries for {len(code_docs)} code files ({max_workers} workers)...")

        summarized = []
        errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all summarization tasks
            future_to_doc = {
                executor.submit(self.summarize_document, doc): doc
                for doc in code_docs
            }

            completed = 0
            for future in as_completed(future_to_doc):
                original_doc = future_to_doc[future]
                completed += 1

                try:
                    summarized_doc = future.result()
                    summarized.append(summarized_doc)
                    if show_progress:
                        file_name = original_doc.metadata.get('file_name', 'unknown')
                        print(f"  [{completed}/{len(code_docs)}] Summarized: {file_name}")
                except Exception as e:
                    # On error, keep the original document
                    summarized.append(original_doc)
                    errors.append((original_doc.metadata.get('file_name', 'unknown'), str(e)))
                    if show_progress:
                        file_name = original_doc.metadata.get('file_name', 'unknown')
                        print(f"  [{completed}/{len(code_docs)}] Error summarizing {file_name}: {e}")

        if show_progress:
            print(f"Summarization complete. {len(code_docs) - len(errors)} succeeded, {len(errors)} failed.")

        # Return summarized code docs + unchanged other docs
        return summarized + other_docs


def summarize_code_documents(
    documents: List[Document],
    model: str = DEFAULT_SUMMARIZER_MODEL,
    show_progress: bool = True
) -> List[Document]:
    """Convenience function to summarize code documents."""
    summarizer = CodeSummarizer(model=model)
    return summarizer.summarize_documents(documents, show_progress)


if __name__ == "__main__":
    # Test the summarizer
    test_code = '''
#include <iostream>
#include <vector>

class MaskChecker {
public:
    bool checkPoint(int x, int y);
    std::vector<bool> checkPoints(const std::vector<Point>& points);
private:
    cv::Mat mask;
};
'''

    summarizer = CodeSummarizer()
    summary = summarizer.summarize_code(test_code, "mask_checker.h", "C++")
    print("Generated Summary:")
    print(summary)
