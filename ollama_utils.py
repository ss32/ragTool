"""
Ollama Utilities Module
Handles model availability checking and automatic downloading.
"""

import subprocess
import sys


class OllamaModelError(Exception):
    """Raised when an Ollama model cannot be found or downloaded."""
    pass


# Cache of models known to be available (session-scoped)
_available_models_cache: set = set()


def get_available_models() -> list:
    """Get list of all locally available Ollama models."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return []

        # Parse the output to find model names
        # Output format: NAME                    ID              SIZE      MODIFIED
        models = []
        lines = result.stdout.strip().split('\n')
        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue
            parts = line.split()
            if parts:
                models.append(parts[0])
        return sorted(models)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def is_model_available(model_name: str) -> bool:
    """Check if a model is available locally in Ollama."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return False

        # Parse the output to find model names
        # Output format: NAME                    ID              SIZE      MODIFIED
        lines = result.stdout.strip().split('\n')
        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue
            # Model name is the first column
            parts = line.split()
            if parts and parts[0] == model_name:
                return True
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def pull_model(model_name: str) -> bool:
    """
    Attempt to pull/download a model from Ollama.

    Returns True if successful, raises OllamaModelError if model not found in repo.
    """
    print(f"Model '{model_name}' not found locally. Attempting to download...")
    print(f"Running: ollama pull {model_name}")
    print("-" * 60)

    try:
        # Run ollama pull with real-time output
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)

        process.wait()
        full_output = ''.join(output_lines)

        if process.returncode != 0:
            # Check for specific error patterns indicating model doesn't exist
            error_patterns = [
                'not found',
                'pull model manifest',
                'file does not exist',
                'invalid model name',
                'unauthorized',
                'no such host',
            ]

            output_lower = full_output.lower()
            for pattern in error_patterns:
                if pattern in output_lower:
                    raise OllamaModelError(
                        f"Model '{model_name}' could not be found in the Ollama repository. "
                        f"Please verify the model name is correct.\n"
                        f"You can browse available models at: https://ollama.com/library"
                    )

            # Generic error
            raise OllamaModelError(
                f"Failed to download model '{model_name}'. "
                f"Please check your internet connection and try again."
            )

        print("-" * 60)
        print(f"Successfully downloaded model: {model_name}")
        return True

    except FileNotFoundError:
        raise OllamaModelError(
            "Ollama is not installed or not in PATH. "
            "Please install Ollama from: https://ollama.com/download"
        )
    except subprocess.TimeoutExpired:
        raise OllamaModelError(
            f"Timeout while downloading model '{model_name}'. "
            "Please try again or check your internet connection."
        )


def ensure_model_available(model_name: str) -> None:
    """
    Ensure a model is available, downloading it if necessary.

    Uses a session-scoped cache to avoid repeated subprocess calls.
    Raises OllamaModelError if the model cannot be made available.
    """
    # Check cache first (fast path)
    if model_name in _available_models_cache:
        return

    if is_model_available(model_name):
        _available_models_cache.add(model_name)
        return

    # Model not available locally, try to pull it
    pull_model(model_name)

    # Verify it's now available
    if not is_model_available(model_name):
        raise OllamaModelError(
            f"Model '{model_name}' was downloaded but is not showing as available. "
            "Please try running 'ollama list' to verify."
        )

    _available_models_cache.add(model_name)


def clear_model_cache():
    """Clear the model availability cache."""
    _available_models_cache.clear()


if __name__ == "__main__":
    # Test the module
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "qwen2.5:7b"

    print(f"Testing model availability: {model}")
    print("=" * 60)

    try:
        ensure_model_available(model)
        print(f"\nModel '{model}' is ready to use.")
    except OllamaModelError as e:
        print(f"\nError: {e}")
        sys.exit(1)
