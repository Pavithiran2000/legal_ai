"""
=============================================================================
Script i: Vector DB Reset & Statistics
=============================================================================

Purpose:
    - Delete ALL documents, chunks, and FAISS vectors from the system
    - Show FAISS index status before and after operations
    - Provide detailed statistics of the vector database

Usage:
    python test_vector_db_reset.py                  # Show stats only (safe)
    python test_vector_db_reset.py --reset           # Delete all data and rebuild
    python test_vector_db_reset.py --stats-only      # Detailed stats only

Requirements:
    - Backend server running on http://localhost:5005
    - pip install httpx tabulate

Architecture Flow:
    This script interacts with:
    ├── GET  /api/admin/faiss/status     → FAISS index stats
    ├── GET  /api/admin/statistics       → System-wide statistics
    ├── GET  /api/admin/documents        → List all documents
    ├── DELETE /api/admin/documents/{id} → Delete individual document
    ├── POST /api/admin/faiss/rebuild    → Rebuild FAISS index
    └── GET  /api/health                 → System health check
=============================================================================
"""
import httpx
import argparse
import sys
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:5005"
TIMEOUT = 60.0


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def check_health():
    """Check if the backend is healthy and all services are running."""
    print_section("System Health Check")
    try:
        r = httpx.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        data = r.json()
        print(f"  Status Code:     {r.status_code}")
        print(f"  Overall Status:  {data.get('status', 'unknown')}")

        components = data.get("components", data)
        if isinstance(components, dict):
            for key, value in components.items():
                if key != "status":
                    icon = "✅" if value in (True, "healthy", "connected", "ok") else "❌"
                    print(f"  {icon} {key}: {value}")
        return r.status_code == 200
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False


def get_faiss_status():
    """Get detailed FAISS index statistics."""
    print_section("FAISS Index Status")
    try:
        r = httpx.get(f"{BASE_URL}/api/admin/faiss/status", timeout=TIMEOUT)
        data = r.json()
        print(f"  Status Code:     {r.status_code}")

        status = data.get("status", data)
        if isinstance(status, dict):
            print(f"  Total Vectors:   {status.get('total_vectors', 'N/A')}")
            print(f"  Dimension:       {status.get('dimension', 'N/A')}")
            print(f"  Index Type:      {status.get('index_type', 'N/A')}")
            partitions = status.get('partitions', {})
            if partitions:
                print(f"  Partitions:      {len(partitions)}")
                for name, count in partitions.items():
                    print(f"    └── {name}: {count} vectors")
            else:
                print(f"  Partitions:      None")
        else:
            print(f"  Response: {json.dumps(data, indent=2)}")
        return data
    except Exception as e:
        print(f"  ❌ Failed to get FAISS status: {e}")
        return None


def get_statistics():
    """Get system-wide statistics."""
    print_section("System Statistics")
    try:
        r = httpx.get(f"{BASE_URL}/api/admin/statistics", timeout=TIMEOUT)
        data = r.json()
        print(f"  Status Code:     {r.status_code}")

        stats = data.get("statistics", data)
        if isinstance(stats, dict):
            for key, value in stats.items():
                label = key.replace("_", " ").title()
                print(f"  {label:30s}: {value}")
        else:
            print(f"  Response: {json.dumps(data, indent=2)}")
        return data
    except Exception as e:
        print(f"  ❌ Failed to get statistics: {e}")
        return None


def list_documents():
    """List all documents in the system."""
    print_section("Document Inventory")
    try:
        r = httpx.get(f"{BASE_URL}/api/admin/documents", timeout=TIMEOUT)
        data = r.json()
        docs = data.get("documents", data if isinstance(data, list) else [])

        if not docs:
            print("  (No documents found)")
            return []

        print(f"  Total Documents: {len(docs)}\n")
        print(f"  {'#':>3} {'ID':>8} {'Filename':>30} {'Status':>10} {'Chunks':>7} {'Partition':>15}")
        print(f"  {'─'*3} {'─'*8} {'─'*30} {'─'*10} {'─'*7} {'─'*15}")

        for i, doc in enumerate(docs, 1):
            doc_id = str(doc.get('id', ''))[:8]
            filename = str(doc.get('original_filename', doc.get('filename', 'N/A')))[:30]
            status = doc.get('status', 'N/A')
            chunks = doc.get('chunk_count', 'N/A')
            partition = doc.get('partition', 'N/A')
            print(f"  {i:3d} {doc_id:>8} {filename:>30} {status:>10} {str(chunks):>7} {partition:>15}")

        return docs
    except Exception as e:
        print(f"  ❌ Failed to list documents: {e}")
        return []


def delete_all_documents(docs):
    """Delete ALL documents from the system."""
    print_section("Deleting All Documents")

    if not docs:
        print("  Nothing to delete.")
        return True

    total = len(docs)
    success = 0
    failed = 0

    for i, doc in enumerate(docs, 1):
        doc_id = doc.get("id")
        filename = doc.get("original_filename", doc.get("filename", "unknown"))
        try:
            r = httpx.delete(f"{BASE_URL}/api/admin/documents/{doc_id}", timeout=TIMEOUT)
            if r.status_code < 300:
                print(f"  ✅ [{i}/{total}] Deleted: {filename} (ID: {doc_id[:8]}...)")
                success += 1
            else:
                print(f"  ❌ [{i}/{total}] Failed:  {filename} → HTTP {r.status_code}: {r.text[:100]}")
                failed += 1
        except Exception as e:
            print(f"  ❌ [{i}/{total}] Error:   {filename} → {e}")
            failed += 1
        time.sleep(0.3)  # Slight delay to avoid overwhelming the server

    print(f"\n  Summary: {success} deleted, {failed} failed out of {total}")
    return failed == 0


def rebuild_faiss_index():
    """Rebuild the FAISS index from database data."""
    print_section("Rebuilding FAISS Index")
    try:
        r = httpx.post(f"{BASE_URL}/api/admin/faiss/rebuild", timeout=120.0)
        data = r.json()
        print(f"  Status Code: {r.status_code}")
        print(f"  Response:    {json.dumps(data, indent=2)}")
        return r.status_code < 300
    except Exception as e:
        print(f"  ❌ Failed to rebuild FAISS index: {e}")
        return False


def run_full_reset():
    """Complete reset: delete all documents and rebuild empty index."""
    print_header("FULL VECTOR DB RESET")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Target:    {BASE_URL}")

    # Pre-reset status
    print_header("PRE-RESET STATUS")
    healthy = check_health()
    if not healthy:
        print("\n  ⚠️  Backend is not healthy. Proceeding anyway...")

    pre_faiss = get_faiss_status()
    pre_stats = get_statistics()
    docs = list_documents()

    if not docs:
        print("\n  ℹ️  No documents to delete. System is already clean.")
        return

    # Confirm
    print(f"\n  ⚠️  WARNING: This will delete {len(docs)} documents and all associated data!")
    confirm = input("  Type 'YES' to confirm: ").strip()
    if confirm != "YES":
        print("  Aborted.")
        return

    # Delete
    delete_all_documents(docs)

    # Rebuild
    rebuild_faiss_index()

    # Post-reset status
    print_header("POST-RESET STATUS")
    get_faiss_status()
    get_statistics()
    remaining = list_documents()

    if not remaining:
        print("\n  ✅ RESET COMPLETE — All documents, chunks, and vectors removed.")
    else:
        print(f"\n  ⚠️  {len(remaining)} documents still remain after reset.")


def run_stats_only():
    """Display detailed statistics without modifying anything."""
    print_header("VECTOR DB STATISTICS REPORT")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Target:    {BASE_URL}")

    check_health()
    get_faiss_status()
    get_statistics()
    list_documents()


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="Vector DB Reset & Statistics Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_vector_db_reset.py               # Show stats (safe, read-only)
  python test_vector_db_reset.py --stats-only   # Detailed stats report
  python test_vector_db_reset.py --reset         # ⚠️  Delete everything + rebuild
        """
    )
    parser.add_argument("--reset", action="store_true", help="Delete ALL documents and rebuild FAISS index")
    parser.add_argument("--stats-only", action="store_true", help="Show detailed statistics only")
    parser.add_argument("--base-url", default=BASE_URL, help=f"Backend URL (default: {BASE_URL})")

    args = parser.parse_args()
    BASE_URL = args.base_url

    if args.reset:
        run_full_reset()
    else:
        run_stats_only()


if __name__ == "__main__":
    main()
