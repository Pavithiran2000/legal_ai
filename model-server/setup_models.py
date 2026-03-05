"""
Setup Ollama models from GGUF files.
Creates Ollama models for both 4B and 8B variants.
"""
import subprocess
import os
import sys
import time


def check_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def create_modelfile(gguf_path: str, modelfile_path: str):
    """Create an Ollama Modelfile for a GGUF model."""
    abs_path = os.path.abspath(gguf_path).replace("\\", "/")
    content = f'''FROM {abs_path}

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>

TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""
'''
    with open(modelfile_path, "w") as f:
        f.write(content)
    print(f"  Created Modelfile: {modelfile_path}")


def create_model(model_name: str, modelfile_path: str):
    """Create an Ollama model from a Modelfile."""
    print(f"  Creating Ollama model '{model_name}'...")
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print(f"  Model '{model_name}' created successfully!")
            return True
        else:
            print(f"  Error creating model: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  Timeout creating model '{model_name}'")
        return False
    except FileNotFoundError:
        print("  ERROR: 'ollama' command not found. Is Ollama installed?")
        return False


def list_models():
    """List available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("\nAvailable Ollama models:")
            print(result.stdout)
    except Exception:
        pass


def setup():
    """Main setup function."""
    print("=" * 60)
    print("  Ollama Model Setup for Sri Lankan Legal AI")
    print("=" * 60)

    # Check Ollama
    if not check_ollama_running():
        print("\nOllama is not running. Starting Ollama...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            time.sleep(5)
            if not check_ollama_running():
                print("ERROR: Could not start Ollama. Please start it manually.")
                return False
        except FileNotFoundError:
            print("ERROR: Ollama not installed. Download from https://ollama.com")
            return False
    print("Ollama is running.")

    models_dir = os.path.dirname(os.path.abspath(__file__))
    success_count = 0

    # Setup 4B model (default)
    gguf_4b = os.path.join(models_dir, "models", "qwen3-4b.gguf")
    if os.path.exists(gguf_4b):
        print(f"\n[1/2] Setting up 4B model...")
        mf_path = os.path.join(models_dir, "Modelfile.4b")
        create_modelfile(gguf_4b, mf_path)
        if create_model("sri-legal-4b", mf_path):
            success_count += 1
    else:
        print(f"\n[1/2] SKIP: 4B GGUF not found at {gguf_4b}")

    # Setup 8B model
    gguf_8b = os.path.join(models_dir, "models", "qwen3_8b.gguf")
    if os.path.exists(gguf_8b):
        print(f"\n[2/2] Setting up 8B model...")
        mf_path = os.path.join(models_dir, "Modelfile.8b")
        create_modelfile(gguf_8b, mf_path)
        if create_model("sri-legal-8b", mf_path):
            success_count += 1
    else:
        print(f"\n[2/2] SKIP: 8B GGUF not found at {gguf_8b}")

    list_models()

    print(f"\nSetup complete: {success_count}/2 models created.")
    return success_count > 0


if __name__ == "__main__":
    setup()
