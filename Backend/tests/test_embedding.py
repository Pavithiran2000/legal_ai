"""
Test Embedding Service — Verify Gemini embeddings work correctly.

Tests:
  1. Single text embedding
  2. Batch embedding
  3. Similarity between related legal texts
  4. Similarity between unrelated texts
  5. Dimension verification

Usage:
    cd Backend
    python scripts/test_embedding.py
"""
import sys
import os
import asyncio
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        result = fn()
        if result:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} — returned False")
            failed += 1
    except Exception as e:
        print(f"  [FAIL] {name} — {e}")
        failed += 1


async def main():
    global passed, failed

    print("\n" + "=" * 60)
    print("  EMBEDDING SERVICE TEST")
    print("=" * 60)

    from src.core.config import settings
    from src.services.embedding_service import EmbeddingService

    service = EmbeddingService()
    await service.initialize()

    actual_dim = service._dimension
    print(f"\n  Embedding model: {settings.embedding_model}")
    print(f"  Configured dimension: {settings.embedding_dimension}")
    print(f"  Actual dimension: {actual_dim}")
    print(f"  Provider: {'Gemini' if service._genai else 'Sentence-Transformers'}")
    print()

    # ── Test 1: Single embedding ──
    print("--- Test 1: Single Text Embedding ---")
    try:
        text = "Wrongful termination of employment in Sri Lanka"
        embedding = await service.embed_query(text)
        assert embedding is not None, "Embedding is None"
        assert len(embedding) == actual_dim, f"Expected dim {actual_dim}, got {len(embedding)}"
        norm = np.linalg.norm(embedding)
        print(f"  [PASS] Single embedding: dim={len(embedding)}, norm={norm:.4f}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Single embedding: {e}")
        failed += 1

    # ── Test 2: Batch embedding ──
    print("\n--- Test 2: Batch Embedding ---")
    try:
        texts = [
            "Employment termination without notice",
            "Gratuity calculation for employees",
            "Industrial dispute resolution process",
        ]
        embeddings = await service.embed_texts(texts)
        assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
        for i, emb in enumerate(embeddings):
            assert len(emb) == actual_dim, f"Embedding {i} has wrong dim"
        print(f"  [PASS] Batch embedding: {len(embeddings)} texts, dim={actual_dim}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Batch embedding: {e}")
        failed += 1

    # ── Test 3: Similarity — related legal texts ──
    print("\n--- Test 3: Similarity Between Related Texts ---")
    try:
        related = [
            "An employee was terminated without any prior notice or compensation",
            "Wrongful dismissal of a worker without following proper termination procedures",
        ]
        emb_related = await service.embed_texts(related)
        vec_a = np.array(emb_related[0])
        vec_b = np.array(emb_related[1])
        # Normalize
        vec_a = vec_a / np.linalg.norm(vec_a)
        vec_b = vec_b / np.linalg.norm(vec_b)
        similarity = float(np.dot(vec_a, vec_b))
        assert similarity > 0.5, f"Related texts should have high similarity, got {similarity:.4f}"
        print(f"  [PASS] Related text similarity: {similarity:.4f} (expected > 0.5)")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Related text similarity: {e}")
        failed += 1

    # ── Test 4: Similarity — unrelated texts ──
    print("\n--- Test 4: Similarity Between Unrelated Texts ---")
    try:
        unrelated = [
            "Wrongful termination of employment under Sri Lankan labour law",
            "Recipe for chicken curry with coconut milk and spices",
        ]
        emb_unrelated = await service.embed_texts(unrelated)
        vec_a = np.array(emb_unrelated[0])
        vec_b = np.array(emb_unrelated[1])
        vec_a = vec_a / np.linalg.norm(vec_a)
        vec_b = vec_b / np.linalg.norm(vec_b)
        similarity = float(np.dot(vec_a, vec_b))
        assert similarity < 0.7, f"Unrelated texts should have low similarity, got {similarity:.4f}"
        print(f"  [PASS] Unrelated text similarity: {similarity:.4f} (expected < 0.7)")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Unrelated text similarity: {e}")
        failed += 1

    # ── Test 5: Dimension consistency ──
    print("\n--- Test 5: Dimension Consistency ---")
    try:
        test_texts = [
            "Short text",
            "A much longer text about the Employment Act and its various provisions regarding employee rights and employer obligations in the context of Sri Lankan employment law",
            "Gratuity",
        ]
        embeddings = await service.embed_texts(test_texts)
        dims = set(len(e) for e in embeddings)
        assert len(dims) == 1, f"Inconsistent dimensions: {dims}"
        the_dim = dims.pop()
        assert the_dim == actual_dim, f"Dimension mismatch: {the_dim} vs {actual_dim}"
        print(f"  [PASS] All {len(test_texts)} texts produce consistent dim={the_dim}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Dimension consistency: {e}")
        failed += 1

    # ── Test 6: FAISS integration check ──
    print("\n--- Test 6: FAISS Index Integration ---")
    try:
        import faiss
        index_path = settings.faiss_index_path
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            assert index.d == actual_dim, f"FAISS dim={index.d} != embedding dim={actual_dim}"
            print(f"  [PASS] FAISS index matches embedding dimension: {index.d}")
            
            # Try a search
            query_emb = np.array([await service.embed_query("wrongful termination")], dtype=np.float32)
            faiss.normalize_L2(query_emb)
            scores, indices = index.search(query_emb, 5)
            print(f"         Top 5 similarity scores: {[f'{s:.4f}' for s in scores[0]]}")
            passed += 1
        else:
            print(f"  [SKIP] No FAISS index found at {index_path}")
            passed += 1
    except Exception as e:
        print(f"  [FAIL] FAISS integration: {e}")
        failed += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    status = "ALL TESTS PASSED" if failed == 0 else "SOME TESTS FAILED"
    print(f"  {status}")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
