"""
=============================================================================
Script ii: Upload Documents — Single File, Bulk, and Folder Upload
=============================================================================

Purpose:
    - Upload individual PDF documents to the backend
    - Bulk upload multiple files at once
    - Upload all PDFs from a specified folder
    - Show upload status, chunk counts, and processing details

Usage:
    python test_upload_docs.py --file ./docs_latest/a5.pdf
    python test_upload_docs.py --files ./docs_latest/a5.pdf ./docs_latest/a6.pdf
    python test_upload_docs.py --folder ./docs_latest
    python test_upload_docs.py --folder ./docs_latest --partition labour_law_v2

Requirements:
    - Backend server running on http://localhost:5005
    - pip install httpx tabulate

Architecture Flow:
    This script interacts with:
    ├── POST /api/admin/documents/upload   → Upload document (multipart/form-data)
    ├── GET  /api/admin/documents          → Verify document listing
    ├── GET  /api/admin/faiss/status       → Check FAISS after upload
    └── GET  /api/admin/statistics         → Check stats after upload

    Upload Pipeline (per file):
    PDF → PyMuPDF extract → paragraph-based chunking (1000 chars, 150 overlap)
    → Gemini embedding (3072 dims) → FAISS IndexFlatIP → PostgreSQL persist
=============================================================================
"""
import httpx
import argparse
import sys
import os
import time
import json
import glob
from pathlib import Path
from datetime import datetime

BASE_URL = "http://localhost:5005"
TIMEOUT = 120.0  # Upload can take time (embedding generation)
SUPPORTED_EXTENSIONS = {".pdf"}


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def upload_single_file(filepath: str, partition: str = None) -> dict:
    """Upload a single document to the backend."""
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        return {"success": False, "file": filepath, "error": "File not found"}

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {"success": False, "file": filepath, "error": f"Unsupported extension: {ext}"}

    filename = os.path.basename(filepath)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"\n  📄 Uploading: {filename} ({file_size_mb:.2f} MB)")

    try:
        start = time.time()

        with open(filepath, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            data = {}
            if partition:
                data["partition"] = partition

            r = httpx.post(
                f"{BASE_URL}/api/admin/documents/upload",
                files=files,
                data=data,
                timeout=TIMEOUT
            )

        elapsed = time.time() - start

        if r.status_code < 300:
            resp = r.json()
            doc = resp.get("document", resp)
            doc_id = doc.get("id", "N/A")
            chunks = doc.get("chunk_count", "N/A")
            status = doc.get("status", "N/A")
            part = doc.get("partition", partition or "default")

            print(f"  ✅ Upload successful ({elapsed:.1f}s)")
            print(f"     Document ID:  {doc_id}")
            print(f"     Chunks:       {chunks}")
            print(f"     Status:       {status}")
            print(f"     Partition:    {part}")

            return {
                "success": True,
                "file": filename,
                "doc_id": str(doc_id),
                "chunks": chunks,
                "status": status,
                "partition": part,
                "time_s": round(elapsed, 2),
                "size_mb": round(file_size_mb, 2),
            }
        else:
            error_text = r.text[:200]
            print(f"  ❌ Upload failed → HTTP {r.status_code}")
            print(f"     Error: {error_text}")
            return {
                "success": False,
                "file": filename,
                "error": f"HTTP {r.status_code}: {error_text}",
                "time_s": round(elapsed, 2),
            }

    except httpx.TimeoutException:
        print(f"  ❌ Upload timed out (>{TIMEOUT}s) — the file may be too large or embedding is slow")
        return {"success": False, "file": filename, "error": "Timeout"}
    except Exception as e:
        print(f"  ❌ Upload error: {e}")
        return {"success": False, "file": filename, "error": str(e)}


def upload_multiple_files(filepaths: list, partition: str = None) -> list:
    """Upload multiple files sequentially."""
    results = []
    total = len(filepaths)

    print_section(f"Uploading {total} File(s)")

    for i, fp in enumerate(filepaths, 1):
        print(f"\n  [{i}/{total}]", end="")
        result = upload_single_file(fp, partition)
        results.append(result)
        if i < total:
            time.sleep(1.0)  # Pause between uploads

    return results


def upload_folder(folder_path: str, partition: str = None) -> list:
    """Upload all PDF files from a folder."""
    folder_path = os.path.abspath(folder_path)

    if not os.path.isdir(folder_path):
        print(f"  ❌ Folder not found: {folder_path}")
        return []

    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))

    if not pdf_files:
        print(f"  ℹ️  No PDF files found in: {folder_path}")
        return []

    print_section(f"Folder Upload: {folder_path}")
    print(f"  Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print(f"    • {os.path.basename(f)}")

    return upload_multiple_files(pdf_files, partition)


def print_results_summary(results: list):
    """Display a summary table of upload results."""
    print_section("Upload Results Summary")

    if not results:
        print("  No uploads attempted.")
        return

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    total_time = sum(r.get("time_s", 0) for r in results)
    total_chunks = sum(r.get("chunks", 0) for r in results if r["success"] and isinstance(r.get("chunks"), int))

    print(f"\n  {'File':>30} {'Status':>10} {'Chunks':>8} {'Time':>8} {'Size MB':>8}")
    print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

    for r in results:
        fname = str(r["file"])[:30]
        status = "✅ OK" if r["success"] else "❌ FAIL"
        chunks = str(r.get("chunks", "-"))
        t = f"{r.get('time_s', 0):.1f}s"
        size = f"{r.get('size_mb', 0):.2f}" if r["success"] else "-"
        print(f"  {fname:>30} {status:>10} {chunks:>8} {t:>8} {size:>8}")

    print(f"\n  Total:    {len(results)} file(s)")
    print(f"  Success:  {success_count}")
    print(f"  Failed:   {fail_count}")
    print(f"  Chunks:   {total_chunks}")
    print(f"  Time:     {total_time:.1f}s")


def show_post_upload_status():
    """Show system status after uploads."""
    print_section("Post-Upload System Status")
    try:
        # FAISS status
        r = httpx.get(f"{BASE_URL}/api/admin/faiss/status", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status", data)
            if isinstance(status, dict):
                print(f"  FAISS Vectors:  {status.get('total_vectors', 'N/A')}")
                print(f"  Dimension:      {status.get('dimension', 'N/A')}")
                partitions = status.get('partitions', {})
                if partitions:
                    for name, count in partitions.items():
                        print(f"    └── {name}: {count} vectors")

        # Statistics
        r = httpx.get(f"{BASE_URL}/api/admin/statistics", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            stats = data.get("statistics", data)
            if isinstance(stats, dict):
                print(f"\n  Total Documents: {stats.get('total_documents', 'N/A')}")
                print(f"  Total Chunks:    {stats.get('total_chunks', 'N/A')}")
                print(f"  Total Queries:   {stats.get('total_queries', 'N/A')}")

    except Exception as e:
        print(f"  ⚠️  Could not fetch post-upload status: {e}")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="Upload Documents to Legal Arise Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_upload_docs.py --file docs_latest/a5.pdf
  python test_upload_docs.py --files docs_latest/a5.pdf docs_latest/a6.pdf docs_latest/a7.pdf
  python test_upload_docs.py --folder docs_latest
  python test_upload_docs.py --folder docs_latest --partition labour_v2
        """
    )
    parser.add_argument("--file", type=str, help="Single file to upload")
    parser.add_argument("--files", nargs="+", type=str, help="Multiple files to upload")
    parser.add_argument("--folder", type=str, help="Upload all PDFs from a folder")
    parser.add_argument("--partition", type=str, default=None, help="Optional partition name for the documents")
    parser.add_argument("--base-url", default=BASE_URL, help=f"Backend URL (default: {BASE_URL})")

    args = parser.parse_args()
    BASE_URL = args.base_url

    print_header("DOCUMENT UPLOAD TOOL")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Target:    {BASE_URL}")

    # Determine upload mode
    results = []

    if args.folder:
        results = upload_folder(args.folder, args.partition)
    elif args.files:
        results = upload_multiple_files(args.files, args.partition)
    elif args.file:
        result = upload_single_file(args.file, args.partition)
        results = [result]
    else:
        # Default: upload from docs_latest if available
        default_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs_latest")
        if os.path.isdir(default_folder):
            print(f"\n  No arguments given. Defaulting to: {default_folder}")
            results = upload_folder(default_folder, args.partition)
        else:
            parser.print_help()
            sys.exit(1)

    # Summary
    print_results_summary(results)
    show_post_upload_status()


if __name__ == "__main__":
    main()
