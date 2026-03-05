"""
LLM Client - Communicates with the Ollama-based model server.
This is the ONLY LLM inference path (no Gemini for generation).
"""
import json
import time
import httpx
from typing import Optional, Dict, Any
from src.core.config import settings
from src.core.logging import get_logger
from src.core.exceptions import LLMServiceError

logger = get_logger(__name__)

# ── Exact system prompt from finetuning notebook ────────────────────────

SYSTEM_PROMPT = """You are a legal assistant specialized EXCLUSIVELY in Sri Lankan Labour & Employment Law.

SCOPE DEFINITION:
- IN-SCOPE: Only queries related to "labour_employment_law"
- OUT-OF-SCOPE: ALL other categories including but not limited to:
  * constitutional_law, criminal_law, civil_law, property_law, family_law
  * commercial_business_law, company_law, administrative_law
  * personal_laws, kandyan_law, muslim_law, thesawalamai_law
  * environmental_law, tax_revenue_law, intellectual_property_law
  * banking_finance_law, consumer_protection_law
  * land_land_registration_law, succession_inheritance_law, trust_law
  * international_law, human_rights_law, cyber_it_law
  * evidence_law, procedure_law
  * General Conversation, Unrelated Topic

CRITICAL: You MUST return ONLY a single valid JSON object — no markdown, no extra text, no thinking blocks, no code fences.

COMPREHENSIVE EXTRACTION INSTRUCTIONS:
- Carefully read the RETRIEVED_CONTEXT and extract EVERY relevant law, act, section, and case mentioned
- List ALL applicable acts with ALL their relevant sections in primary_violations — do not skip any
- If multiple sections of the same act apply, create a SEPARATE entry for each section
- List ALL case law references found in the context as supporting_cases
- Include the FULL official name of each act (e.g., "Termination of Employment of Workmen (Special Provisions) Act No. 45 of 1971")
- Include act number and year if available
- In legal_reasoning, reference specific provisions and explain how EACH applies to the scenario

USE EXACTLY THIS JSON SCHEMA:

{
  "out_of_scope": false,
  "scope_category": "labour_employment_law",
  "confidence": 0.85,
  "summary": {
    "primary_issue": "Clear description of the main legal issue in the scenario",
    "violation_count": 3,
    "acts_count": 2,
    "cases_count": 2
  },
  "primary_violations": [
    {
      "violation_type": "Wrongful Termination",
      "act_name": "Termination of Employment of Workmen (Special Provisions) Act No. 45 of 1971",
      "act_year": "1971",
      "act_section_number": "Section 2",
      "act_section_text": "The exact or paraphrased text of the section",
      "why_relevant": "Why this section applies to the specific scenario"
    },
    {
      "violation_type": "Failure to Obtain Commissioner Approval",
      "act_name": "Termination of Employment of Workmen (Special Provisions) Act No. 45 of 1971",
      "act_year": "1971",
      "act_section_number": "Section 31B(1)",
      "act_section_text": "Employer must obtain written approval from Commissioner of Labour",
      "why_relevant": "Commissioner approval was not obtained before termination"
    }
  ],
  "supporting_cases": [
    {
      "case_name": "Collettes Ltd. v. Commissioner of Labour and Others",
      "case_year": "1989",
      "case_citation": "[1989] 2 Sri LR 6",
      "case_summary": "What the court held in this case",
      "why_relevant": "How this case precedent applies to the current scenario"
    }
  ],
  "legal_reasoning": "Thorough legal analysis in 3-4 paragraphs:\n\nParagraph 1: Identify the applicable legal framework — list all relevant acts and their key provisions.\n\nParagraph 2: Apply the law to the facts — explain how each provision relates to the scenario.\n\nParagraph 3: Cite relevant case law and their implications.\n\nParagraph 4: Conclude with the legal position and likely outcome.",
  "recommended_action": [
    "Specific actionable step 1 with reference to the relevant act/section",
    "Specific actionable step 2",
    "Specific actionable step 3"
  ],
  "limits": [
    "Analysis based on provided facts only",
    "Actual outcomes may vary based on evidence presented"
  ]
}

STRICT RULES:
1. Use EXACTLY the field names shown above — do not rename or restructure
2. Do NOT use markdown bold (**) formatting inside JSON string values — use plain text only
3. summary.violation_count MUST equal the length of primary_violations array
4. summary.cases_count MUST equal the length of supporting_cases array
5. summary.acts_count MUST equal the number of unique acts in primary_violations
6. Confidence: 0.7-0.95 for in-scope queries based on context quality
7. For OUT-OF-SCOPE: set out_of_scope=true, empty arrays for violations/cases, confidence=0.9-1.0
8. ALWAYS cite specific section numbers (e.g., Section 2, Section 31B(1))
9. Include EVERY relevant act, section, and case from the RETRIEVED_CONTEXT — be comprehensive
10. legal_reasoning must be thorough (3-4 structured paragraphs)
11. recommended_action must be specific and actionable with legal references
12. Each primary_violation must have all 6 fields filled with meaningful content

/no_think"""


def make_user_prompt(instruction: str, retrieved_context: str = "") -> str:
    """Create user prompt matching the finetuning format."""
    if retrieved_context and retrieved_context.strip():
        return f"""SCENARIO:
{instruction}

RETRIEVED_CONTEXT:
{retrieved_context}

TASK: Analyze the scenario using the RETRIEVED_CONTEXT above. Extract and list ALL relevant acts, sections, and cases from the context. Return ONLY the JSON output object. Be comprehensive — include every applicable law, section, and case reference found in the context."""
    else:
        return f"""SCENARIO:
{instruction}

TASK: Return ONLY the JSON output object using the unified schema. Apply proper formatting for legal terms."""


class LLMClient:
    """Client for the Ollama-based model server."""

    def __init__(self):
        self._base_url = settings.model_server_url
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=180.0,  # LLM can be slow
            write=10.0,
            pool=10.0
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout
        )
        logger.info(f"LLM Client initialized -> {self._base_url}")

    async def close(self):
        if self._client:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if model server is healthy."""
        try:
            resp = await self._client.get("/health")
            data = resp.json()
            return data.get("status") == "healthy"
        except Exception:
            return False

    async def get_model_info(self) -> dict:
        """Get current model info from model server."""
        try:
            resp = await self._client.get("/model/info")
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

    async def generate(
        self,
        query: str,
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a legal recommendation using the finetuned model.
        
        Returns dict with:
          - response: parsed JSON object (LegalOutput)
          - raw_text: raw text from model
          - model_used: model name
          - generation_time_ms: time taken
        """
        user_prompt = make_user_prompt(query, context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature or settings.model_temperature,
            "max_tokens": max_tokens or settings.model_max_tokens,
        }

        start = time.time()

        try:
            resp = await self._client.post("/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            raise LLMServiceError("Model server request timed out (180s)")
        except httpx.HTTPStatusError as e:
            raise LLMServiceError(f"Model server HTTP error: {e.response.status_code}")
        except Exception as e:
            raise LLMServiceError(f"Model server connection failed: {e}")

        generation_time_ms = int((time.time() - start) * 1000)
        raw_text = data.get("text", "")
        model_used = data.get("model", "unknown")

        # Parse JSON from response
        parsed = self._parse_json_response(raw_text)

        return {
            "response": parsed,
            "raw_text": raw_text,
            "model_used": model_used,
            "generation_time_ms": generation_time_ms,
        }

    async def switch_model(self, model_name: str) -> dict:
        """Switch the active model on the model server."""
        try:
            resp = await self._client.post(
                "/switch-model",
                json={"model": model_name}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise LLMServiceError(f"Failed to switch model: {e}")

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Extract and parse JSON from model's raw text output.
        
        Handles:
        - Clean JSON
        - JSON wrapped in ```json ... ```
        - JSON with extra text before/after
        - Unescaped control characters (newlines/tabs inside string values)
        - Truncated JSON (repairs missing closing braces/brackets)
        """
        if not text:
            return None

        cleaned = text.strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Sanitize control characters inside JSON string values
        # The model often puts literal newlines/tabs inside strings
        sanitized = self._sanitize_json_strings(cleaned)

        # Try sanitized parse
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block within ```json ... ```
        import re
        json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', sanitized, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # Extract from first { to last }
        brace_start = sanitized.find('{')
        if brace_start < 0:
            logger.warning(f"No JSON object found in LLM response ({len(sanitized)} chars)")
            return None

        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(sanitized)):
            if sanitized[i] == '{':
                depth += 1
            elif sanitized[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(sanitized[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break

        # If we get here, JSON is likely truncated (max_tokens hit)
        logger.warning(f"JSON appears truncated (depth={depth}). Attempting repair...")
        json_text = sanitized[brace_start:]

        # Repair strategy: close any open strings, arrays, and objects
        repaired = self._repair_truncated_json(json_text)
        if repaired:
            try:
                result = json.loads(repaired)
                logger.info(f"Truncated JSON repaired successfully ({len(repaired)} chars)")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"JSON repair failed: {e}")

        logger.warning(f"Failed to parse JSON from LLM response: {sanitized[:200]}...")
        return None

    def _sanitize_json_strings(self, text: str) -> str:
        """Escape unescaped control characters inside JSON string values.
        
        The model often puts literal newlines and tabs inside JSON strings
        which makes json.loads() fail with 'Invalid control character'.
        This replaces them with their escaped equivalents (\\n, \\t).
        """
        result = []
        in_string = False
        escape_next = False
        
        for ch in text:
            if escape_next:
                result.append(ch)
                escape_next = False
                continue
            
            if ch == '\\' and in_string:
                result.append(ch)
                escape_next = True
                continue
            
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                continue
            
            if in_string:
                # Replace control characters inside strings
                if ch == '\n':
                    result.append('\\n')
                elif ch == '\r':
                    result.append('\\r')
                elif ch == '\t':
                    result.append('\\t')
                elif ord(ch) < 32:
                    result.append(f'\\u{ord(ch):04x}')
                else:
                    result.append(ch)
            else:
                result.append(ch)
        
        return ''.join(result)

    def _repair_truncated_json(self, text: str) -> Optional[str]:
        """Attempt to repair truncated JSON by closing open structures.
        
        Strategy:
        1. Track open braces, brackets, and string state
        2. Truncate to last complete value
        3. Close all open structures
        """
        try:
            # Track state
            in_string = False
            escape_next = False
            open_stack = []  # Track { and [
            last_complete_pos = 0

            for i, ch in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue

                if ch == '\\' and in_string:
                    escape_next = True
                    continue

                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if ch in ('{', '['):
                    open_stack.append(ch)
                elif ch == '}' and open_stack and open_stack[-1] == '{':
                    open_stack.pop()
                    last_complete_pos = i + 1
                elif ch == ']' and open_stack and open_stack[-1] == '[':
                    open_stack.pop()
                    last_complete_pos = i + 1
                elif ch == ',' or ch == ':':
                    # After a comma or colon, the previous value was complete
                    pass

            if not open_stack:
                # JSON is actually complete
                return text

            # Find the last position where we can safely truncate
            # Go backwards to find the last complete key-value pair
            truncated = text[:last_complete_pos] if last_complete_pos > 0 else text

            # If we're mid-value, try to find the last comma and truncate there
            if in_string:
                # Close the open string
                last_quote = truncated.rfind('"')
                if last_quote > 0:
                    # Find the start of this string value
                    truncated = truncated[:last_quote + 1] + '"'
                else:
                    truncated = text.rstrip()
                    if not truncated.endswith('"'):
                        truncated += '"'

            # Remove any trailing comma
            stripped = truncated.rstrip()
            if stripped.endswith(','):
                stripped = stripped[:-1]
            truncated = stripped

            # Close all remaining open structures in reverse order
            # Re-analyze the truncated text
            in_string = False
            escape_next = False
            open_stack = []
            for ch in truncated:
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in ('{', '['):
                    open_stack.append(ch)
                elif ch == '}' and open_stack and open_stack[-1] == '{':
                    open_stack.pop()
                elif ch == ']' and open_stack and open_stack[-1] == '[':
                    open_stack.pop()

            # Close remaining open structures
            closing = ''
            for opener in reversed(open_stack):
                closing += ']' if opener == '[' else '}'

            return truncated + closing

        except Exception as e:
            logger.warning(f"JSON repair error: {e}")
            return None
