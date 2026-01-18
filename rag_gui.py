#!/usr/bin/env python3
"""
RAG Tool - Interactive GUI
A Gradio-based web interface for querying RAG databases.
"""

import os
from pathlib import Path
from typing import Generator, List, Tuple

import gradio as gr

from query_engine import QueryEngine, DEFAULT_LLM_MODEL
from rag_database import DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTOR_WEIGHT
from ollama_utils import get_available_models
from reranker import DEFAULT_RERANK_MODEL


# Global query engine instance (lazily initialized)
_query_engine = None
_current_db_name = None


def get_available_databases() -> List[str]:
    """Get list of available RAG databases."""
    rag_tool_dir = Path.home() / ".rag_tool"
    databases = []

    if rag_tool_dir.exists():
        for item in rag_tool_dir.iterdir():
            if item.is_dir() and (item / "chroma.sqlite3").exists():
                databases.append(item.name)

    # Add default if it exists
    default_path = Path(DEFAULT_DB_PATH)
    if default_path.exists() and (default_path / "chroma.sqlite3").exists():
        if "chromadb" not in databases:
            databases.append("chromadb (default)")

    return sorted(databases) if databases else ["No databases found"]


def get_db_path(db_name: str) -> str:
    """Convert database name to full path."""
    if db_name == "chromadb (default)" or db_name == "chromadb":
        return DEFAULT_DB_PATH
    return os.path.expanduser(f"~/.rag_tool/{db_name}")


def init_query_engine(
    db_name: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    enable_hybrid: bool = True,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    enable_reranking: bool = False,
    rerank_model: str = DEFAULT_RERANK_MODEL
) -> Tuple[bool, str]:
    """Initialize or reinitialize the query engine."""
    global _query_engine, _current_db_name

    if db_name == "No databases found":
        return False, "No databases available. Please create one using the CLI."

    db_path = get_db_path(db_name)

    try:
        _query_engine = QueryEngine(
            db_path=db_path,
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_model=embedding_model,
            llm_model=llm_model,
            warm_up=True,
            enable_response_cache=True,
            enable_hybrid=enable_hybrid,
            vector_weight=vector_weight,
            enable_reranking=enable_reranking,
            rerank_model=rerank_model
        )
        _current_db_name = db_name

        stats = _query_engine.db.get_stats()
        hybrid_status = "hybrid" if stats.get('hybrid_search') else "vector-only"
        return True, f"Connected to '{db_name}' ({stats['document_count']} chunks, {hybrid_status})"
    except Exception as e:
        return False, f"Error connecting to database: {str(e)}"


def get_database_stats(db_name: str) -> str:
    """Get statistics for the selected database."""
    global _query_engine, _current_db_name

    if db_name == "No databases found":
        return "No database selected"

    # Initialize if needed
    if _query_engine is None or _current_db_name != db_name:
        success, msg = init_query_engine(db_name)
        if not success:
            return msg

    try:
        db_stats = _query_engine.db.get_stats()
        cache_stats = _query_engine.get_cache_stats()

        stats_text = f"""Database Statistics
==================
Collection: {db_stats['collection_name']}
Document Chunks: {db_stats['document_count']}
Database Path: {db_stats['db_path']}
Embedding Model: {db_stats['embedding_model']}

Cache Statistics
================
Embedding Cache Size: {cache_stats['cache_size']}/{cache_stats['max_size']}
Embedding Cache Hit Rate: {cache_stats['hit_rate']}
Response Cache Size: {cache_stats['response_cache_size']}/{cache_stats['response_cache_max']}
Response Cache Hit Rate: {cache_stats['response_cache_hit_rate']}
"""
        return stats_text
    except Exception as e:
        return f"Error getting stats: {str(e)}"


def clear_cache(db_name: str) -> str:
    """Clear all caches for the current database."""
    global _query_engine, _current_db_name

    if db_name == "No databases found":
        return "No database selected"

    # Initialize if needed
    if _query_engine is None or _current_db_name != db_name:
        success, msg = init_query_engine(db_name)
        if not success:
            return msg

    try:
        _query_engine.db.clear_embedding_cache()
        _query_engine.clear_response_cache()
        return "Caches cleared successfully!"
    except Exception as e:
        return f"Error clearing cache: {str(e)}"


def strip_thinking_tags(text: str) -> str:
    """Remove <thinking>...</thinking> and <think>...</think> tags from text."""
    import re
    # Remove <thinking>...</thinking> tags (including multiline)
    text = re.sub(r'<thinking>.*?</thinking>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove <think>...</think> tags (including multiline)
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Also handle unclosed tags during streaming (remove from opening tag to end)
    text = re.sub(r'<thinking>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def chat_response(
    message: str,
    history: List[Tuple[str, str]],
    db_name: str,
    llm_model: str,
    n_results: int,
    use_streaming: bool,
    enable_hybrid: bool = True,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    enable_reranking: bool = False
) -> Generator[str, None, None]:
    """Generate chat response with optional streaming."""
    global _query_engine, _current_db_name

    if not message.strip():
        yield "Please enter a question."
        return

    if db_name == "No databases found":
        yield "No database selected. Please create a database first using:\n`python rag_tool.py create --input /path/to/docs --name mydb`"
        return

    # Initialize or reinitialize if database changed
    if _query_engine is None or _current_db_name != db_name:
        success, msg = init_query_engine(
            db_name, llm_model,
            enable_hybrid=enable_hybrid,
            vector_weight=vector_weight,
            enable_reranking=enable_reranking
        )
        if not success:
            yield msg
            return

    # Check if LLM model changed
    if _query_engine.llm_model != llm_model:
        success, msg = init_query_engine(
            db_name, llm_model,
            enable_hybrid=enable_hybrid,
            vector_weight=vector_weight,
            enable_reranking=enable_reranking
        )
        if not success:
            yield msg
            return

    # Update search settings if they changed
    _query_engine.enable_hybrid = enable_hybrid
    _query_engine.vector_weight = vector_weight
    _query_engine.enable_reranking = enable_reranking

    try:
        if use_streaming:
            response_text = ""
            for token in _query_engine.query_stream(
                message,
                n_results=n_results,
                show_sources=True
            ):
                response_text += token
                # Strip thinking tags and yield cleaned response
                cleaned = strip_thinking_tags(response_text)
                if cleaned:
                    yield cleaned
        else:
            response = _query_engine.query(
                message,
                n_results=n_results,
                show_sources=True
            )
            yield strip_thinking_tags(response)
    except Exception as e:
        yield f"Error generating response: {str(e)}"


def on_database_change(db_name: str, llm_model: str) -> Tuple[str, str]:
    """Handle database selection change."""
    if db_name == "No databases found":
        return "No database selected", ""

    success, msg = init_query_engine(db_name, llm_model)
    stats = get_database_stats(db_name) if success else ""
    return msg, stats


def refresh_databases():
    """Refresh the list of available databases."""
    databases = get_available_databases()
    return gr.update(choices=databases, value=databases[0] if databases else None)


def refresh_models():
    """Refresh the list of available LLM models."""
    models = get_available_models()
    if not models:
        models = [DEFAULT_LLM_MODEL]
    return gr.update(choices=models, value=models[0] if models else None)


def create_gui(
    default_llm_model: str = DEFAULT_LLM_MODEL,
    share: bool = False,
    server_port: int = 7860
) -> gr.Blocks:
    """Create and return the Gradio interface."""

    # Get initial database list
    databases = get_available_databases()
    initial_db = databases[0] if databases else "No databases found"

    # Get initial LLM models list
    llm_models = get_available_models()
    # Ensure default model is in list, or use first available
    if default_llm_model in llm_models:
        initial_llm = default_llm_model
    elif llm_models:
        initial_llm = llm_models[0]
    else:
        llm_models = [default_llm_model]
        initial_llm = default_llm_model

    with gr.Blocks(title="RAG Query Tool") as demo:
        gr.Markdown("""
        # RAG Query Tool
        Ask questions about your documents using local LLMs via Ollama.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat",
                    height="67vh"  # 2/3 of viewport height
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about your documents...",
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                clear_chat_btn = gr.Button("Clear Chat", size="sm")

            with gr.Column(scale=1):
                # Settings panel
                gr.Markdown("### Settings")

                db_dropdown = gr.Dropdown(
                    choices=databases,
                    value=initial_db,
                    label="Select Database",
                    interactive=True
                )
                refresh_btn = gr.Button("Refresh Databases", size="sm")

                status_text = gr.Textbox(
                    label="Status",
                    value="Select a database to begin",
                    interactive=False
                )

                llm_model_dropdown = gr.Dropdown(
                    choices=llm_models,
                    value=initial_llm,
                    label="LLM Model",
                    interactive=True
                )
                refresh_models_btn = gr.Button("Refresh Models", size="sm")

                n_results_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Context Documents"
                )

                streaming_checkbox = gr.Checkbox(
                    label="Stream Response",
                    value=True
                )

                gr.Markdown("### Search Settings")

                hybrid_checkbox = gr.Checkbox(
                    label="Hybrid Search (BM25 + Vector)",
                    value=True
                )

                vector_weight_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_VECTOR_WEIGHT,
                    step=0.1,
                    label="Vector Weight (BM25 = 1 - this)"
                )

                reranking_checkbox = gr.Checkbox(
                    label="Enable Reranking (slower, better quality)",
                    value=False
                )

                gr.Markdown("### Database Info")
                stats_text = gr.Textbox(
                    label="Statistics",
                    value="",
                    interactive=False,
                    lines=12
                )

                with gr.Row():
                    clear_cache_btn = gr.Button("Clear Cache", size="sm")
                    refresh_stats_btn = gr.Button("Refresh Stats", size="sm")

                cache_status = gr.Textbox(
                    label="Cache Status",
                    value="",
                    interactive=False,
                    visible=True
                )

        # Event handlers
        def user_message(message, history):
            """Add user message to history."""
            if history is None:
                history = []
            history = history + [gr.ChatMessage(role="user", content=message)]
            return "", history

        def bot_response(history, db_name, llm_model, n_results, use_streaming,
                         enable_hybrid, vector_weight, enable_reranking):
            """Generate bot response."""
            if not history:
                return history

            # Get user message from last entry
            last_entry = history[-1]
            if hasattr(last_entry, "content"):
                user_msg = last_entry.content
            elif isinstance(last_entry, dict):
                user_msg = last_entry.get("content", "")
            else:
                user_msg = str(last_entry)

            # Handle case where content might be a list (multimodal)
            if isinstance(user_msg, list):
                # Extract text from list items
                text_parts = []
                for item in user_msg:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                user_msg = " ".join(text_parts) if text_parts else ""

            # Convert history to old format for chat_response function
            old_history = []
            for i in range(0, len(history) - 1, 2):
                if i + 1 < len(history):
                    h1, h2 = history[i], history[i + 1]
                    c1 = h1.content if hasattr(h1, "content") else (h1.get("content", "") if isinstance(h1, dict) else str(h1))
                    c2 = h2.content if hasattr(h2, "content") else (h2.get("content", "") if isinstance(h2, dict) else str(h2))
                    old_history.append((c1, c2))

            # Generate response
            for response in chat_response(
                user_msg, old_history, db_name, llm_model, n_results, use_streaming,
                enable_hybrid, vector_weight, enable_reranking
            ):
                # Yield history with assistant response
                yield history + [gr.ChatMessage(role="assistant", content=response)]

        # Submit message
        msg_input.submit(
            user_message,
            [msg_input, chatbot],
            [msg_input, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, db_dropdown, llm_model_dropdown, n_results_slider, streaming_checkbox,
             hybrid_checkbox, vector_weight_slider, reranking_checkbox],
            chatbot
        )

        submit_btn.click(
            user_message,
            [msg_input, chatbot],
            [msg_input, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, db_dropdown, llm_model_dropdown, n_results_slider, streaming_checkbox,
             hybrid_checkbox, vector_weight_slider, reranking_checkbox],
            chatbot
        )

        # Clear chat
        clear_chat_btn.click(lambda: None, None, chatbot, queue=False)

        # Database selection change
        db_dropdown.change(
            on_database_change,
            [db_dropdown, llm_model_dropdown],
            [status_text, stats_text]
        )

        # Refresh databases
        refresh_btn.click(
            refresh_databases,
            None,
            db_dropdown
        )

        # Refresh LLM models
        refresh_models_btn.click(
            refresh_models,
            None,
            llm_model_dropdown
        )

        # Clear cache
        clear_cache_btn.click(
            clear_cache,
            [db_dropdown],
            [cache_status]
        )

        # Refresh stats
        refresh_stats_btn.click(
            get_database_stats,
            [db_dropdown],
            [stats_text]
        )

        # Initialize on load
        demo.load(
            on_database_change,
            [db_dropdown, llm_model_dropdown],
            [status_text, stats_text]
        )

    return demo


def launch_gui(
    llm_model: str = DEFAULT_LLM_MODEL,
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1"
):
    """Launch the Gradio GUI."""
    demo = create_gui(default_llm_model=llm_model, share=share, server_port=server_port)
    demo.queue()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Tool GUI")
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"Ollama LLM model (default: {DEFAULT_LLM_MODEL})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1, use 0.0.0.0 for all interfaces)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL for sharing"
    )

    args = parser.parse_args()

    print(f"Starting RAG Tool GUI on http://{args.host}:{args.port}")
    launch_gui(
        llm_model=args.llm_model,
        share=args.share,
        server_port=args.port,
        server_name=args.host
    )
