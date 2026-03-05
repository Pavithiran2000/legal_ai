"""
Document Service - PDF parsing and text chunking.
"""
import os
import re
from typing import List, Tuple
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Handles PDF parsing and text chunking."""

    def __init__(self):
        self._upload_dir = settings.upload_dir
        os.makedirs(self._upload_dir, exist_ok=True)

    async def save_upload(self, filename: str, content: bytes) -> str:
        """Save uploaded file and return path."""
        filepath = os.path.join(self._upload_dir, filename)
        with open(filepath, "wb") as f:
            f.write(content)
        logger.info(f"Saved upload: {filename} ({len(content)} bytes)")
        return filepath

    def extract_text(self, filepath: str) -> str:
        """Extract text from a PDF file."""
        import fitz  # PyMuPDF

        text_parts = []
        try:
            doc = fitz.open(filepath)
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
            doc.close()
        except Exception as e:
            logger.error(f"PDF extraction failed for {filepath}: {e}")
            raise ValueError(f"Failed to extract text from PDF: {e}")

        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} chars from {os.path.basename(filepath)}")
        return full_text

    def chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap

        if not text or not text.strip():
            return []

        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If paragraph itself is too long, split it
                if len(para) > chunk_size:
                    sub_chunks = self._split_long_text(para, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Start new chunk with overlap from previous
                    if chunks and chunk_overlap > 0:
                        overlap_text = chunks[-1][-chunk_overlap:]
                        current_chunk = f"{overlap_text}\n\n{para}"
                    else:
                        current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Filter out tiny chunks
        chunks = [c for c in chunks if len(c) >= 50]

        logger.info(f"Created {len(chunks)} chunks (avg {sum(len(c) for c in chunks) // max(len(chunks), 1)} chars)")
        return chunks

    def _split_long_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split a long paragraph by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 <= chunk_size:
                current = f"{current} {sent}" if current else sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def detect_partition(self, filename: str, text: str) -> str:
        """Detect document partition/category based on filename and content."""
        filename_lower = filename.lower()

        partition_keywords = {
            "employment": ["employment", "employer", "employee", "hire", "hiring"],
            "termination": ["termination", "dismissal", "retrenchment", "redundancy"],
            "wages": ["wages", "salary", "remuneration", "payment", "minimum wage"],
            "industrial_disputes": ["industrial dispute", "trade union", "collective", "strike"],
            "safety": ["safety", "health", "occupational", "hazard", "factory"],
            "maternity": ["maternity", "pregnancy", "maternal"],
            "gratuity": ["gratuity", "severance", "retirement benefit"],
            "epf_etf": ["epf", "etf", "provident fund", "trust fund"],
            "general": [],
        }

        # Check filename first
        for partition, keywords in partition_keywords.items():
            if any(kw in filename_lower for kw in keywords):
                return partition

        # Check content
        text_lower = text[:2000].lower()
        best_match = "general"
        best_count = 0
        for partition, keywords in partition_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_match = partition

        return best_match
