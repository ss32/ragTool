# Local RAG Tool

A local RAG (Retrieval-Augmented Generation) tool that creates vector databases from documents and lets you query them using Ollama LLMs. Similar to NotebookLM, but fully local.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Create a database from your documents
python3 rag_tool.py create --input /path/to/docs --name mydb

# Query it
python3 rag_tool.py query "What does this code do?" --name mydb

# Or use the web GUI
python3 rag_tool.py gui
```

## Commands

### Create Database

```bash
python3 rag_tool.py create --input /path/to/docs --name mydb
```

Options:
- `--summarize`, `-s` — Generate LLM summaries for code files (slower but better quality)
- `--workers N` — Parallel file loading
- `--fresh` — Start fresh, ignore previous progress

### Query

```bash
python3 rag_tool.py query "Your question" --name mydb
```

Options:
- `--stream` — Stream response tokens
- `--llm-model MODEL` — Use a different LLM (default: qwen3:8b)
- `--results N` — Number of context documents (default: 5)

### Interactive Mode

```bash
python3 rag_tool.py interactive --name mydb
```

### Web GUI

```bash
python3 rag_tool.py gui --host 0.0.0.0 --port 8000
```

Features a chat interface with database/model selection dropdowns and streaming responses.

## Supported File Types

| Type | Extensions |
|------|------------|
| Documents | `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html` |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` |
| Text | `.txt`, `.md`, `.rst` |
| Code | `.py`, `.js`, `.ts`, `.cpp`, `.c`, `.java`, `.go`, `.rs`, `.rb`, `.sh`, `.json`, `.yaml`, etc. |

## Models

Models are auto-downloaded on first use. Defaults:

| Purpose | Model |
|---------|-------|
| Embeddings | `all-minilm:33m` |
| LLM | `qwen3:8b` |

Change with `--embedding-model` or `--llm-model`.

## Tips

- Use `--summarize` when ingesting source code without documentation
- Databases are stored in `~/.rag_tool/<name>/`
- Ingestion is resumable — interrupt with Ctrl+C and run the same command to continue
