# Local RAG + Query

This tool uses `ollama` to create a vectorized database from an input directory containing text files, PDFs, source code, Markdown, etc. The database can then be queried for questions similar to NotebookLM, but everything is local, including the database.

## Setup

`python3 -m pip install -r requirements.txt`

## Usage

The most basic usage requires two steps

1. `python3 rag_tool.py create --input /path/to/input/directory`
2. `python3 rag_tool.py query "What does this code do?"`

It is recommended that you name databases for querying later:

```bash
python3 rag_tool.py create --name myDB --input /path/to/input/directory`
```

The same `--name` arg is used to query the database later

```bash
python3 rag_tool.py --name myDB query "What do?"
```

**Note:** The default mode for building the database optmizes speed over quality. The following examples cover a deeper but slower method for building the RAG database

Pass the `--summarize` argument to pass the files to an LLM for context generation. This is particularly helpful if your input directory contains only source code and no documentation. The LLM will parse the code and make its best guess as to the functionality and use. This extra context is extremely useful when querying.

### Choose a Model

Every step of the process can be tuned to fit your compute budget. Pass the name of an [ollama model](https://ollama.com/) to use it for summary or querying.

* Use `gpt-oss:20b` as a summary model using the `--summarize-model` arg

```bash
python3 rag_tool.py create -i /path/to/input --name myDB --summarize --summarize-model gpt-oss:20b
```

* Use `qwen2.5:7b` to query an existing database using the `--llm` arg

```bash
 python3 rag_tool.py --name myDB query "Summarize what this code does" --llm qwen2.5:7b