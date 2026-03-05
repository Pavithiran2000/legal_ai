"""
=============================================================================
Script iii: List Documents with Chunk Relations
=============================================================================

Purpose:
    - List all documents in the system with detailed metadata
    - Show chunk counts, chunk content previews, and relationships
    - Display document → chunk → FAISS vector mapping
    - Verify data integrity between PostgreSQL and FAISS

Usage:
    python test_list_documents.py                     # List all documents
    python test_list_documents.py --details           # Include chunk previews
    python test_list_documents.py --doc-id <uuid>     # Show specific document
    python test_list_documents.py --verify            # Verify DB ↔ FAISS consistency

Requirements:
    - Backend server running on http://localhost:5005
    - pip install httpx

Architecture Flow:
    This script interacts with:
    ├── GET  /api/admin/documents          → List all documents with metadata
    ├── GET  /api/admin/faiss/status       → FAISS vector counts per partition
    ├── GET  /api/admin/statistics         → Overall system statistics
    └── GET  /api/health/ready             → Readiness check
=============================================================================
"""
import httpx
import argparse
import sys
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


def check_readiness():
    """Check if backend is ready."""
    try:
        r = httpx.get(f"{BASE_URL}/api/health/ready", timeout=TIMEOUT)
        data = r.json()
        ready = data.get("ready", data.get("status") == "ok")
        print(f"  Backend Ready: {'✅ Yes' if ready else '❌ No'}")
        return ready
    except Exception as e:
        print(f"  ❌ Cannot connect to backend: {e}")
        return False


def list_all_documents(show_details: bool = False):
    """List all documents with their metadata and chunk information."""
    print_section("Document Inventory")
    try:
        r = httpx.get(f"{BASE_URL}/api/admin/documents", timeout=TIMEOUT)
        data = r.json()
        docs = data.get("documents", data if isinstance(data, list) else [])

        if not docs:
            print("  (No documents in the system)")
            return []

        print(f"  Total Documents: {len(docs)}\n")

        for i, doc in enumerate(docs, 1):
            doc_id = doc.get("id", "N/A")
            filename = doc.get("original_filename", doc.get("filename", "N/A"))
            status = doc.get("status", "N/A")
            chunk_count = doc.get("chunk_count", "N/A")
            partition = doc.get("partition", "default")
            created = doc.get("created_at", "N/A")
            file_size = doc.get("file_size", None)
            content_hash = doc.get("content_hash", None)

            print(f"  Document #{i}")
            print(f"  ┌─────────────────────────────────────────────────")
            print(f"  │ ID:           {doc_id}")
            print(f"  │ Filename:     {filename}")
            print(f"  │ Status:       {status}")
            print(f"  │ Chunks:       {chunk_count}")
            print(f"  │ Partition:    {partition}")
            print(f"  │ Created:      {created}")
            if file_size is not None:
                print(f"  │ File Size:    {file_size} bytes")
            if content_hash:
                print(f"  │ Content Hash: {content_hash[:16]}...")

            # Show chunk details if available and requested
            chunks = doc.get("chunks", [])
            if show_details and chunks:
                print(f"  │")
                print(f"  │ Chunk Details ({len(chunks)} chunks):")
                for j, chunk in enumerate(chunks, 1):
                    chunk_id = str(chunk.get("id", ""))[:8]
                    content = chunk.get("content", "")
                    preview = content[:80].replace("\n", " ") + "..." if len(content) > 80 else content.replace("\n", " ")
                    char_count = len(content)
                    page = chunk.get("page_number", chunk.get("metadata", {}).get("page", "N/A"))
                    position = chunk.get("chunk_index", chunk.get("position", j))
                    has_embedding = chunk.get("has_embedding", chunk.get("embedding_id") is not None)

                    print(f"  │   Chunk {j:3d} │ ID: {chunk_id} │ Pos: {position} │ Page: {page} │ Chars: {char_count} │ Embedded: {'✅' if has_embedding else '❌'}")
                    print(f"  │            │ Preview: {preview}")
            elif show_details and not chunks:
                print(f"  │")
                print(f"  │ (Chunk details not available in listing response)")

            print(f"  └─────────────────────────────────────────────────")

        return docs

    except Exception as e:
        print(f"  ❌ Failed to list documents: {e}")
        return []


def show_document_detail(doc_id: str):
    """Show detailed information about a specific document."""
    print_section(f"Document Detail: {doc_id[:16]}...")
    try:
        # Try direct document endpoint
        r = httpx.get(f"{BASE_URL}/api/admin/documents/{doc_id}", timeout=TIMEOUT)
        if r.status_code == 200:
            doc = r.json()
            if isinstance(doc, dict):
                doc_data = doc.get("document", doc)
                print(json.dumps(doc_data, indent=2, default=str))
        else:
            # Fall back to listing and filter
            r = httpx.get(f"{BASE_URL}/api/admin/documents", timeout=TIMEOUT)
            data = r.json()
            docs = data.get("documents", data if isinstance(data, list) else [])
            found = [d for d in docs if str(d.get("id", "")) == doc_id]
            if found:
                print(json.dumps(found[0], indent=2, default=str))
            else:
                print(f"  Document not found: {doc_id}")

    except Exception as e:
        print(f"  ❌ Failed to get document detail: {e}")


def verify_consistency():
    """Verify consistency between PostgreSQL documents/chunks and FAISS vectors."""
    print_section("Data Consistency Verification")

    # Get document listing
    try:
        r = httpx.get(f"{BASE_URL}/api/admin/documents", timeout=TIMEOUT)
        docs_data = r.json()
        docs = docs_data.get("documents", docs_data if isinstance(docs_data, list) else [])
        total_docs = len(docs)
        total_db_chunks = sum(doc.get("chunk_count", 0) for doc in docs if isinstance(doc.get("chunk_count"), int))
    except Exception as e:
        print(f"  ❌ Cannot fetch documents: {e}")
        return

    # Get FAISS status
    try:
        r = httpx.get(f"{BASE_URL}/api/admin/faiss/status", timeout=TIMEOUT)
        faiss_data = r.json()
        faiss_status = faiss_data.get("status", faiss_data)
        total_vectors = faiss_status.get("total_vectors", 0) if isinstance(faiss_status, dict) else 0
    except Exception as e:
        print(f"  ❌ Cannot fetch FAISS status: {e}")
        total_vectors = "ERROR"

    # Get statistics
    try:
        r = httpx.get(f"{BASE_URL}/api/admin/statistics", timeout=TIMEOUT)
        stats_data = r.json()
        stats = stats_data.get("statistics", stats_data)
        stats_docs = stats.get("total_documents", "N/A") if isinstance(stats, dict) else "N/A"
        stats_chunks = stats.get("total_chunks", "N/A") if isinstance(stats, dict) else "N/A"
    except Exception as e:
        stats_docs = "ERROR"
        stats_chunks = "ERROR"

    # Report
    print(f"\n  Source              │ Documents │ Chunks/Vectors")
    print(f"  ────────────────────┼───────────┼────────────────")
    print(f"  Document Listing    │ {total_docs:>9} │ {total_db_chunks:>14}")
    print(f"  Statistics API      │ {str(stats_docs):>9} │ {str(stats_chunks):>14}")
    print(f"  FAISS Index         │     -     │ {str(total_vectors):>14}")

    # Consistency checks
    print(f"\n  Consistency Checks:")
    issues = 0

    # Check: DB chunks ≈ FAISS vectors
    if isinstance(total_vectors, int):
        if total_db_chunks == total_vectors:
            print(f"  ✅ DB chunks ({total_db_chunks}) match FAISS vectors ({total_vectors})")
        else:
            diff = abs(total_db_chunks - total_vectors)
            print(f"  ⚠️  DB chunks ({total_db_chunks}) ≠ FAISS vectors ({total_vectors}) — diff: {diff}")
            issues += 1

    # Check: Listing matches statistics
    if isinstance(stats_docs, int):
        if total_docs == stats_docs:
            print(f"  ✅ Document count consistent ({total_docs})")
        else:
            print(f"  ⚠️  Listing ({total_docs}) ≠ Statistics ({stats_docs})")
            issues += 1

    if isinstance(stats_chunks, int) and isinstance(total_db_chunks, int):
        if total_db_chunks == stats_chunks:
            print(f"  ✅ Chunk count consistent ({total_db_chunks})")
        else:
            print(f"  ⚠️  Listed chunks ({total_db_chunks}) ≠ Statistics chunks ({stats_chunks})")
            issues += 1

    # Check: All docs have chunks
    zero_chunk_docs = [d for d in docs if d.get("chunk_count", 0) == 0]
    if zero_chunk_docs:
        print(f"  ⚠️  {len(zero_chunk_docs)} document(s) have 0 chunks:")
        for d in zero_chunk_docs:
            print(f"      └── {d.get('original_filename', d.get('filename', 'unknown'))}")
        issues += 1
    else:
        print(f"  ✅ All documents have chunks")

    # Check: Document statuses
    statuses = {}
    for d in docs:
        s = d.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\n  Document Status Distribution:")
    for s, count in sorted(statuses.items()):
        icon = "✅" if s in ("processed", "completed", "ready", "active") else "⚠️"
        print(f"    {icon} {s}: {count}")

    # Partition summary
    partitions = {}
    for d in docs:
        p = d.get("partition", "default")
        partitions[p] = partitions.get(p, 0) + 1
    if partitions:
        print(f"\n  Partition Distribution:")
        for p, count in sorted(partitions.items()):
            print(f"    📁 {p}: {count} document(s)")

    if issues == 0:
        print(f"\n  ✅ ALL CONSISTENCY CHECKS PASSED")
    else:
        print(f"\n  ⚠️  {issues} potential issue(s) found")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="List Documents with Chunk Relations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_list_documents.py                  # List all documents
  python test_list_documents.py --details        # Include chunk previews
  python test_list_documents.py --doc-id <uuid>  # Show specific document
  python test_list_documents.py --verify         # Verify DB ↔ FAISS consistency
        """
    )
    parser.add_argument("--details", action="store_true", help="Show chunk content previews")
    parser.add_argument("--doc-id", type=str, help="Show details for a specific document ID")
    parser.add_argument("--verify", action="store_true", help="Verify data consistency")
    parser.add_argument("--base-url", default=BASE_URL, help=f"Backend URL (default: {BASE_URL})")

    args = parser.parse_args()
    BASE_URL = args.base_url

    print_header("DOCUMENT LISTING & CHUNK RELATIONS")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Target:    {BASE_URL}")

    check_readiness()

    if args.doc_id:
        show_document_detail(args.doc_id)
    elif args.verify:
        list_all_documents(show_details=False)
        verify_consistency()
    else:
        list_all_documents(show_details=args.details)


if __name__ == "__main__":
    main()
