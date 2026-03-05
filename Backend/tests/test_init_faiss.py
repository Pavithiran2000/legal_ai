"""
=============================================================================
Script 0: FAISS Index Initialisation
=============================================================================

Purpose:
    - Check whether the FAISS index directories and files exist on disk
    - If missing, create the required directories and call the backend
      rebuild endpoint to initialise an empty (or data-populated) index
    - Safe to run at any time — no data is deleted

Usage:
    python tests/test_init_faiss.py                # Check status + auto-init if missing
    python tests/test_init_faiss.py --check        # Check status only (no changes)
    python tests/test_init_faiss.py --force        # Re-initialise even if index exists

Requirements:
    - Backend server running on http://localhost:5005
    - pip install httpx

Architecture Flow:
    GET  /api/health                  → Confirm backend is up
    GET  /api/admin/faiss/status      → Read current index state
    POST /api/admin/faiss/rebuild     → Rebuild / initialise index from stored chunks
    GET  /api/admin/faiss/status      → Confirm final state
=============================================================================
"""
import httpx
import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime

BASE_URL = "http://localhost:5005"
TIMEOUT = 120.0

# Paths relative to the Backend root (where the server is launched from)
FAISS_INDEX_DIR = Path("./models/faiss_index")
FAISS_PARTITIONS_DIR = Path("./models/faiss_partitions")
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
FAISS_IDS_FILE = FAISS_INDEX_DIR / "chunk_ids.npy"


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def status_line(label: str, value, ok: bool = True):
    icon = "✔" if ok else "✘"
    print(f"  [{icon}] {label:<35} {value}")


# ── 1. Health check ───────────────────────────────────────────────────────────

def check_health() -> bool:
    print_section("Backend Health")
    try:
        r = httpx.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        data = r.json()
        alive = r.status_code == 200
        status_line("Backend status", data.get("status", "unknown"), alive)
        return alive
    except Exception as e:
        status_line("Backend reachable", f"ERROR — {e}", False)
        return False


# ── 2. Local filesystem check ─────────────────────────────────────────────────

def check_local_dirs() -> dict:
    print_section("Local Directory Check")
    result = {
        "index_dir": FAISS_INDEX_DIR.exists(),
        "partitions_dir": FAISS_PARTITIONS_DIR.exists(),
        "index_file": FAISS_INDEX_FILE.exists(),
        "ids_file": FAISS_IDS_FILE.exists(),
    }
    status_line("models/faiss_index/    exists", result["index_dir"], result["index_dir"])
    status_line("models/faiss_partitions/ exists", result["partitions_dir"], result["partitions_dir"])
    status_line("index.faiss            exists", result["index_file"], result["index_file"])
    status_line("chunk_ids.npy          exists", result["ids_file"], result["ids_file"])
    return result


def create_dirs():
    print_section("Creating Missing Directories")
    for d in [FAISS_INDEX_DIR, FAISS_PARTITIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        status_line(str(d), "created / already exists", True)


# ── 3. API FAISS status ───────────────────────────────────────────────────────

def get_faiss_status() -> dict:
    try:
        r = httpx.get(f"{BASE_URL}/api/admin/faiss/status", timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def print_faiss_status(label: str, data: dict):
    print_section(label)
    total = data.get("total_vectors", data.get("vector_count", "N/A"))
    dim   = data.get("dimension", "N/A")
    parts = data.get("partitions", {})
    indexed = data.get("indexed", data.get("is_loaded", total != 0))
    status_line("Total vectors", total, total not in (0, "N/A"))
    status_line("Dimension", dim, dim not in (0, "N/A"))
    status_line("Loaded / initialised", indexed, bool(indexed))
    if isinstance(parts, dict) and parts:
        print(f"\n  Partitions:")
        for name, info in parts.items():
            count = info if isinstance(info, int) else info.get("vector_count", info.get("vectors", "?"))
            print(f"    • {name:<25} {count} vectors")
    elif isinstance(parts, list) and parts:
        print(f"\n  Partitions: {', '.join(parts)}")


# ── 4. Rebuild / initialise ───────────────────────────────────────────────────

def trigger_init() -> dict:
    print_section("Triggering FAISS Rebuild / Init")
    print("  Calling POST /api/admin/faiss/rebuild …")
    t0 = time.time()
    try:
        r = httpx.post(f"{BASE_URL}/api/admin/faiss/rebuild", timeout=TIMEOUT)
        elapsed = (time.time() - t0) * 1000
        data = r.json() if r.status_code in (200, 201) else {}
        ok = r.status_code in (200, 201)
        status_line("HTTP status", r.status_code, ok)
        status_line("Duration", f"{elapsed:.0f} ms", True)
        if ok:
            vecs = data.get("total_vectors", data.get("vectors", "?"))
            status_line("Vectors after rebuild", vecs, True)
        else:
            print(f"  [WARN]  Response: {str(r.text)[:200]}")
        return data
    except Exception as e:
        status_line("Rebuild request", f"FAILED — {e}", False)
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Initialise FAISS index if not found."
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check status only — make no changes (overrides default auto-init)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild the index even if it already exists"
    )
    args = parser.parse_args()

    print_header(f"FAISS INDEX INIT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Backend : {BASE_URL}")
    print(f"  Mode    : {'check-only' if args.check else ('force rebuild' if args.force else 'auto-init if missing')}")

    # 1. Backend must be up
    if not check_health():
        print("\n  [ERROR] Backend is not reachable. Start the server first.\n")
        sys.exit(1)

    # 2. Local dirs
    local = check_local_dirs()
    dirs_missing = not local["index_dir"] or not local["partitions_dir"]
    files_missing = not local["index_file"] or not local["ids_file"]

    # 3. Current API status
    before = get_faiss_status()
    print_faiss_status("Current FAISS Status (Before)", before)

    already_loaded = bool(before.get("total_vectors", 0) or before.get("is_loaded"))

    # ── Decision ─────────────────────────────────────────────────────
    if args.check:
        # Read-only — just report and exit
        print("\n")
        if already_loaded and not files_missing:
            print("  ✔  FAISS index is present and loaded. Nothing to do.")
        else:
            print("  ✘  FAISS index is missing or uninitialised.")
            print("     Run without --check to auto-initialise.")
        print()
        sys.exit(0 if (already_loaded and not files_missing) else 1)

    needs_init = args.force or dirs_missing or files_missing or not already_loaded

    if not needs_init:
        print("\n  ✔  FAISS index already exists and is loaded — nothing to do.")
        print("     Use --force to rebuild anyway.\n")
        sys.exit(0)

    # 4. Create directories if needed
    if dirs_missing:
        create_dirs()

    # 5. Trigger rebuild / init via API
    trigger_init()

    # 6. Verify
    after = get_faiss_status()
    print_faiss_status("FAISS Status After Init", after)

    after_loaded = bool(after.get("total_vectors") is not None)
    after_vecs   = after.get("total_vectors", 0)

    print_section("Summary")
    if after_loaded:
        if after_vecs == 0:
            status_line("Result", "Index initialised (empty — upload docs next)", True)
        else:
            status_line("Result", f"Index rebuilt with {after_vecs} vectors", True)
        print()
        sys.exit(0)
    else:
        status_line("Result", "Init may have failed — check server logs", False)
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
