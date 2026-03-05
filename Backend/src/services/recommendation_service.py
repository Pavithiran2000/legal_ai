"""
Recommendation Service - Main pipeline orchestrator.
Query → Embed → FAISS search → Build context → LLM → Parse → Return
"""
import json
import uuid
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.logging import get_logger
from src.core.exceptions import LLMServiceError, OutOfScopeError
from src.services.embedding_service import EmbeddingService
from src.services.faiss_service import FAISSService
from src.services.llm_client import LLMClient
from src.repositories.chunk_repo import ChunkRepository
from src.repositories.query_repo import QueryRepository
from src.schemas.query import (
    LegalOutput,
    OutputSummary,
    PrimaryViolation,
    SupportingCase,
    QueryResponse,
)

logger = get_logger(__name__)


class RecommendationService:
    """Orchestrates the full recommendation pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        faiss_service: FAISSService,
        llm_client: LLMClient,
    ):
        self._embedding = embedding_service
        self._faiss = faiss_service
        self._llm = llm_client

    async def get_recommendation(
        self,
        query_text: str,
        db: AsyncSession,
        top_k: int = None,
        temperature: float = None,
    ) -> QueryResponse:
        """
        Full pipeline:
        1. Embed query
        2. FAISS similarity search (wide net)
        3. Document-diverse reranking
        4. Build context string (capped at max length)
        5. Send to finetuned model
        6. Parse and validate response
        7. Save to database
        """
        # Use a wide initial search to find chunks across ALL relevant documents
        initial_search_k = max((top_k or settings.top_k) * 3, 40)
        final_k = top_k or settings.top_k

        # Step 1: Embed question
        logger.info(f"Processing query: {query_text[:80]}...")
        query_embedding = await self._embedding.embed_query(query_text)

        # Step 2: FAISS search (wide net with lower threshold)
        search_results = await self._faiss.search(
            query_embedding,
            top_k=initial_search_k,
            min_similarity=0.20,  # Lower threshold for wider coverage
        )
        logger.info(f"FAISS returned {len(search_results)} raw results (searching top {initial_search_k})")

        # Step 3: Retrieve chunk texts and apply document-diverse reranking
        chunk_repo = ChunkRepository(db)
        
        # Group results by document for diversity
        doc_chunks: dict[str, list] = {}  # doc_id -> [(score, chunk)]
        for chunk_id, score in search_results:
            chunk = await chunk_repo.get_by_id(chunk_id)
            if chunk:
                doc_id = chunk.document_id
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append((score, chunk))
        
        logger.info(f"Chunks span {len(doc_chunks)} documents")
        
        # Document-diverse selection:
        # Round 1: Pick the BEST chunk from each document (ensures coverage)
        # Round 2: Fill remaining slots with next-best chunks by score
        selected = []
        used_chunk_ids = set()
        
        # Round 1 — best chunk per document (sorted by best score per doc)
        doc_best = []
        for doc_id, chunks in doc_chunks.items():
            best_score, best_chunk = max(chunks, key=lambda x: x[0])
            doc_best.append((best_score, best_chunk, doc_id))
        doc_best.sort(key=lambda x: x[0], reverse=True)
        
        for score, chunk, doc_id in doc_best:
            if len(selected) >= final_k:
                break
            selected.append((score, chunk))
            used_chunk_ids.add(chunk.id)
        
        # Round 2 — fill remaining with best remaining chunks across all docs
        if len(selected) < final_k:
            remaining = []
            for doc_id, chunks in doc_chunks.items():
                for score, chunk in chunks:
                    if chunk.id not in used_chunk_ids:
                        remaining.append((score, chunk))
            remaining.sort(key=lambda x: x[0], reverse=True)
            
            for score, chunk in remaining:
                if len(selected) >= final_k:
                    break
                selected.append((score, chunk))
                used_chunk_ids.add(chunk.id)
        
        logger.info(f"Selected {len(selected)} chunks after diversity reranking")
        
        # Build context parts (sorted by score, capped at max context length)
        selected.sort(key=lambda x: x[0], reverse=True)
        context_parts = []
        total_chars = 0
        max_context = settings.rag_max_context_length
        
        for score, chunk in selected:
            entry = f"[Relevance: {score:.3f}]\n{chunk.content}"
            if total_chars + len(entry) > max_context and context_parts:
                logger.info(f"Context capped at {total_chars} chars ({len(context_parts)} chunks)")
                break
            context_parts.append(entry)
            total_chars += len(entry)

        context_str = "\n\n---\n\n".join(context_parts) if context_parts else ""
        logger.info(f"Final context: {len(context_parts)} chunks, {len(context_str)} chars")

        # Step 4: Send to LLM
        try:
            llm_result = await self._llm.generate(
                query=query_text,
                context=context_str,
                temperature=temperature,
            )
        except LLMServiceError as e:
            logger.error(f"LLM generation failed: {e}")
            raise

        # Step 5: Parse response
        parsed = llm_result.get("response")
        raw_text = llm_result.get("raw_text", "")
        model_used = llm_result.get("model_used", "unknown")
        generation_time_ms = llm_result.get("generation_time_ms", 0)

        logger.info(f"LLM raw_text length: {len(raw_text)}, first 500 chars: {raw_text[:500]}")
        logger.info(f"LLM parsed: {type(parsed)} - keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")
        
        # Debug: log the full parsed data to diagnose empty responses
        if parsed and isinstance(parsed, dict):
            logger.info(f"PARSED violations count: {len(parsed.get('primary_violations', []))}")
            logger.info(f"PARSED cases count: {len(parsed.get('supporting_cases', []))}")
            logger.info(f"PARSED confidence: {parsed.get('confidence')}")
            logger.info(f"PARSED legal_reasoning length: {len(str(parsed.get('legal_reasoning', '')))}")
            if not parsed.get("primary_violations") and not parsed.get("legal_reasoning"):
                logger.warning(f"FULL PARSED DATA: {json.dumps(parsed, ensure_ascii=False)[:2000]}")
                logger.warning(f"FULL RAW TEXT: {raw_text[:3000]}")

        legal_output = self._build_legal_output(parsed) if parsed else None
        logger.info(f"LegalOutput built: {legal_output is not None}")

        # Step 6: Save query to database
        query_repo = QueryRepository(db)
        query_record = await query_repo.create(
            query_text=query_text,
            response_json=parsed or {"error": "Failed to parse response", "raw": raw_text[:2000]},
            out_of_scope=legal_output.out_of_scope if legal_output else None,
            scope_category=legal_output.scope_category if legal_output else None,
            confidence=legal_output.confidence if legal_output else None,
            model_used=model_used,
            generation_time_ms=generation_time_ms,
            context_chunks_used=len(context_parts),
        )

        # Step 7: Build response
        from datetime import datetime, timezone
        return QueryResponse(
            query_id=str(query_record.id),
            query=query_text,
            recommendation=legal_output if legal_output else LegalOutput(),
            model_used=model_used,
            generation_time_ms=generation_time_ms,
            context_chunks_used=len(context_parts),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _strip_bold(text: str) -> str:
        """Remove markdown bold markers (**) from text values.
        
        The finetune dataset uses plain text without bold markers.
        The model sometimes adds **bold** markers despite instructions not to.
        """
        if isinstance(text, str):
            return text.replace("**", "").strip()
        return text

    def _build_legal_output(self, data: dict) -> Optional[LegalOutput]:
        """Build a validated LegalOutput from raw parsed JSON.
        
        Handles multiple response formats since the model doesn't always
        follow the exact expected schema.
        """
        try:
            # ── Detect out_of_scope ──
            out_of_scope = data.get("out_of_scope", False)
            scope_category = self._strip_bold(data.get("scope_category", "labour_employment_law"))

            # ── Build summary ──
            summary_data = data.get("summary", {})
            if isinstance(summary_data, str):
                summary_data = {"primary_issue": summary_data}
            elif not isinstance(summary_data, dict):
                summary_data = {}

            # ── Extract violations (flexible key matching) ──
            violations_raw = (
                data.get("primary_violations")
                or data.get("violations")
                or data.get("legal_issues")
                or []
            )

            # If model nests violations under legal_basis.acts
            if not violations_raw and isinstance(data.get("legal_basis"), dict):
                lb = data["legal_basis"]
                acts = lb.get("acts", [])
                violations_raw = []
                for act in acts:
                    if isinstance(act, dict):
                        sections = act.get("sections", [])
                        for sec in (sections if sections else [act]):
                            violations_raw.append({
                                "violation_type": sec.get("text", sec.get("definition", "")),
                                "act_name": act.get("name", ""),
                                "act_year": act.get("number", ""),
                                "act_section_number": sec.get("section", sec.get("number", "")),
                                "act_section_text": sec.get("text", sec.get("definition", "")),
                                "why_relevant": sec.get("relevance", ""),
                            })
                    elif isinstance(act, str):
                        violations_raw.append(act)
            elif not violations_raw and isinstance(data.get("legal_basis"), list):
                violations_raw = data["legal_basis"]

            violations = []
            for v in violations_raw:
                if isinstance(v, dict):
                    violations.append(PrimaryViolation(
                        violation_type=self._strip_bold(str(v.get("violation_type", v.get("type", v.get("name", ""))))),
                        act_name=self._strip_bold(str(v.get("act_name", v.get("act", v.get("name", ""))))),
                        act_year=self._strip_bold(str(v.get("act_year", v.get("year", v.get("number", ""))))),
                        act_section_number=self._strip_bold(str(v.get("act_section_number", v.get("section", "")))),
                        act_section_text=self._strip_bold(str(v.get("act_section_text", v.get("text", "")))),
                        why_relevant=self._strip_bold(str(v.get("why_relevant", v.get("relevance", "")))),
                    ))
                elif isinstance(v, str):
                    violations.append(PrimaryViolation(violation_type=self._strip_bold(v), act_name=""))

            # ── Extract cases (flexible key matching) ──
            cases_raw = (
                data.get("supporting_cases")
                or data.get("cases")
                or data.get("relevant_cases")
                or data.get("case_law")
                or []
            )
            cases = []
            for c in cases_raw:
                if isinstance(c, dict):
                    cases.append(SupportingCase(
                        case_name=self._strip_bold(str(c.get("case_name", c.get("name", c.get("title", ""))))),
                        case_year=self._strip_bold(str(c.get("case_year", c.get("year", "")))),
                        case_citation=self._strip_bold(str(c.get("case_citation", c.get("citation", "")))),
                        case_summary=self._strip_bold(str(c.get("case_summary", c.get("summary", c.get("description", ""))))),
                        why_relevant=self._strip_bold(str(c.get("why_relevant", c.get("relevance", "")))),
                    ))
                elif isinstance(c, str):
                    cases.append(SupportingCase(case_name=self._strip_bold(c)))

            # ── Extract legal reasoning (flexible) ──
            legal_reasoning = (
                data.get("legal_reasoning")
                or data.get("reasoning")
                or data.get("analysis")
                or data.get("conclusion")
                or ""
            )
            if isinstance(legal_reasoning, list):
                legal_reasoning = "\n\n".join(str(item) for item in legal_reasoning)
            elif isinstance(legal_reasoning, dict):
                legal_reasoning = json.dumps(legal_reasoning)
            legal_reasoning = self._strip_bold(str(legal_reasoning))

            # ── Extract recommended actions (flexible) ──
            rec_action = (
                data.get("recommended_action")
                or data.get("recommendations")
                or data.get("actions")
                or data.get("rights")
                or []
            )
            if isinstance(rec_action, str):
                rec_action = [self._strip_bold(rec_action)]
            elif isinstance(rec_action, list):
                flat = []
                for item in rec_action:
                    if isinstance(item, str):
                        flat.append(self._strip_bold(item))
                    elif isinstance(item, dict):
                        flat.append(self._strip_bold(str(item.get("description", item.get("action", json.dumps(item))))))
                rec_action = flat

            # ── Extract limits ──
            limits = data.get("limits", data.get("limitations", []))
            if isinstance(limits, str):
                limits = [self._strip_bold(limits)]
            elif isinstance(limits, list):
                limits = [self._strip_bold(str(item)) for item in limits]

            # ── Confidence ──
            confidence = float(data.get("confidence", 0.0))
            if confidence == 0.0 and not out_of_scope and (violations or cases or legal_reasoning):
                confidence = 0.75

            # ── Summary from violations/cases if not provided ──
            primary_issue = self._strip_bold(summary_data.get("primary_issue", ""))
            if not primary_issue and legal_reasoning:
                primary_issue = str(legal_reasoning)[:200] + ("..." if len(str(legal_reasoning)) > 200 else "")

            violation_count = summary_data.get("violation_count", len(violations))
            acts_count = summary_data.get("acts_count", len(set(v.act_name for v in violations if v.act_name)))
            cases_count = summary_data.get("cases_count", len(cases))

            summary = OutputSummary(
                primary_issue=primary_issue,
                violation_count=violation_count,
                acts_count=acts_count,
                cases_count=cases_count,
            )

            return LegalOutput(
                out_of_scope=out_of_scope,
                scope_category=scope_category,
                summary=summary,
                primary_violations=violations,
                supporting_cases=cases,
                legal_reasoning=str(legal_reasoning),
                recommended_action=rec_action,
                limits=limits,
                confidence=confidence,
            )
        except Exception as e:
            logger.error(f"Failed to build LegalOutput: {e}", exc_info=True)
            return None
