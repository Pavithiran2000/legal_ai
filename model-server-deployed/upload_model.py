"""
Chunked Upload of GGUF Model to Modal Volume.

Splits the 8B model (~4.7 GB) into 300 MB chunks and uploads each
to the Modal volume 'qwen-3-finetuned' under /model-data/chunks/.
After all chunks are uploaded, triggers a remote merge on Modal.

Usage:
    python upload_model.py                           # Upload + merge
    python upload_model.py --check                   # Check volume status
    python upload_model.py --merge-only              # Merge existing chunks
    python upload_model.py --source <path_to_gguf>   # Custom source path
"""
import os
import sys
import math
import time
import hashlib
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────

VOLUME_NAME = "qwen-3-finetuned"
MODEL_DIR = "/model-data"               # Mount path inside Modal containers
DEFAULT_SOURCE = os.path.join(
    os.path.dirname(__file__), "..", "model-server", "models", "qwen3_8b.gguf"
)
GGUF_FILENAME = "qwen3_8b.gguf"
CHUNK_SIZE_MB = 300                      # 300 MB per chunk
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024


# ── Helpers ──────────────────────────────────────────────────────────

def md5_file(path: str) -> str:
    """Return hex MD5 of a file (for integrity)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_file(source: str, out_dir: str, chunk_bytes: int = CHUNK_SIZE_BYTES):
    """
    Split *source* into numbered chunks in *out_dir*.
    Returns list of (chunk_path, chunk_size) tuples.
    """
    os.makedirs(out_dir, exist_ok=True)
    file_size = os.path.getsize(source)
    n_chunks = math.ceil(file_size / chunk_bytes)
    logger.info(f"Splitting {source} ({file_size / 1e9:.2f} GB) into {n_chunks} chunks of {chunk_bytes / 1e6:.0f} MB")

    chunks = []
    with open(source, "rb") as f:
        for i in range(n_chunks):
            chunk_name = f"chunk_{i:04d}"
            chunk_path = os.path.join(out_dir, chunk_name)
            data = f.read(chunk_bytes)
            with open(chunk_path, "wb") as cf:
                cf.write(data)
            chunks.append((chunk_path, len(data)))
            logger.info(f"  [{i + 1}/{n_chunks}] {chunk_name}  ({len(data) / 1e6:.1f} MB)")

    return chunks


# ── Upload ───────────────────────────────────────────────────────────

def upload_chunks(source_path: str):
    """Stream-split model one chunk at a time, upload each, then trigger merge."""
    import modal
    import tempfile

    if not os.path.exists(source_path):
        logger.error(f"Source file not found: {source_path}")
        sys.exit(1)

    file_size = os.path.getsize(source_path)
    n_chunks = math.ceil(file_size / CHUNK_SIZE_BYTES)
    logger.info(f"Source : {source_path}")
    logger.info(f"Size   : {file_size / 1e9:.2f} GB")
    logger.info(f"Volume : {VOLUME_NAME}")
    logger.info(f"Chunk  : {CHUNK_SIZE_MB} MB  ({n_chunks} chunks)")

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    logger.info("=" * 60)
    logger.info("  Uploading chunks to Modal volume (streaming)")
    logger.info("=" * 60)

    total_uploaded = 0
    with open(source_path, "rb") as src:
        for idx in range(n_chunks):
            chunk_name = f"chunk_{idx:04d}"
            remote_path = f"chunks/{chunk_name}"

            # Write one chunk to a temp file
            tmp_path = os.path.join(tempfile.gettempdir(), f"modal_{chunk_name}")
            data = src.read(CHUNK_SIZE_BYTES)
            chunk_size = len(data)

            with open(tmp_path, "wb") as cf:
                cf.write(data)

            # Upload to Modal
            t0 = time.time()
            with vol.batch_upload(force=True) as batch:
                batch.put_file(tmp_path, remote_path)
            elapsed = time.time() - t0

            # Remove temp file immediately
            os.remove(tmp_path)

            speed = (chunk_size / 1e6) / max(elapsed, 0.01)
            total_uploaded += chunk_size
            pct = total_uploaded / file_size * 100
            logger.info(
                f"  [{idx + 1}/{n_chunks}] {chunk_name} "
                f"→ {elapsed:.1f}s ({speed:.1f} MB/s)  [{pct:.0f}%]"
            )

    logger.info(f"All {n_chunks} chunks uploaded ({total_uploaded / 1e9:.2f} GB)")

    # Trigger remote merge
    logger.info("=" * 60)
    logger.info("  Triggering remote merge on Modal …")
    logger.info("=" * 60)
    trigger_merge()


def trigger_merge():
    """Call the merge_model_chunks function deployed in modal_app.py."""
    import modal

    merge_fn = modal.Function.from_name("qwen3-legal-model-server", "merge_model_chunks")
    result = merge_fn.remote(GGUF_FILENAME)
    logger.info(f"Merge result: {result}")
    return result


def check_volume():
    """List contents of the Modal volume."""
    import modal

    list_fn = modal.Function.from_name("qwen3-legal-model-server", "list_volume_contents")
    contents = list_fn.remote()
    logger.info("Volume contents:")
    total = 0
    for item in contents:
        logger.info(f"  {item['path']:40s}  {item['size_mb']:>10.1f} MB")
        total += item["size_mb"]
    logger.info(f"  {'TOTAL':40s}  {total:>10.1f} MB")

    check_fn = modal.Function.from_name("qwen3-legal-model-server", "check_model_ready")
    status = check_fn.remote()
    logger.info(f"Model ready: {status}")


def upload_direct(source_path: str):
    """
    Fallback: Upload the entire GGUF directly (no chunking).
    Only use if chunked upload fails.
    """
    import modal

    if not os.path.exists(source_path):
        logger.error(f"Source not found: {source_path}")
        sys.exit(1)

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    file_size = os.path.getsize(source_path)
    logger.info(f"Direct upload: {source_path} ({file_size / 1e9:.2f} GB)")

    t0 = time.time()
    with vol.batch_upload(force=True) as batch:
        batch.put_file(source_path, GGUF_FILENAME)
    elapsed = time.time() - t0
    logger.info(f"Upload complete in {elapsed:.1f}s ({file_size / 1e6 / max(elapsed, 1):.1f} MB/s)")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload GGUF model to Modal volume (chunked)")
    parser.add_argument(
        "--source", type=str, default=DEFAULT_SOURCE,
        help=f"Path to GGUF file (default: {DEFAULT_SOURCE})"
    )
    parser.add_argument("--check", action="store_true", help="Check volume status only")
    parser.add_argument("--merge-only", action="store_true", help="Trigger merge without uploading")
    parser.add_argument("--direct", action="store_true", help="Upload entire file directly (no chunking)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_MB, help="Chunk size in MB (default: 300)")

    args = parser.parse_args()

    global CHUNK_SIZE_BYTES
    CHUNK_SIZE_BYTES = args.chunk_size * 1024 * 1024

    if args.check:
        check_volume()
    elif args.merge_only:
        trigger_merge()
    elif args.direct:
        upload_direct(os.path.abspath(args.source))
    else:
        upload_chunks(os.path.abspath(args.source))


if __name__ == "__main__":
    main()
