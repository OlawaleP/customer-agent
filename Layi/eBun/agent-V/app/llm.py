"""
Model client.

Points at any OpenAI-compatible chat completions endpoint, which covers:
  - vLLM  (self-hosted, e.g. Qwen2.5-72B-Instruct)
  - Ollama (self-hosted, local/small scale)
  - OpenRouter / Together (hosted open-weight models, no GPU needed)
  - OpenAI itself, if you ever want to A/B against a closed model

Set these in .env:
  LLM_BASE_URL   e.g. http://localhost:8000/v1   (vLLM)  or  http://localhost:11434/v1  (Ollama)
  LLM_API_KEY    dummy value is fine for local vLLM/Ollama; real key for OpenRouter/OpenAI
  LLM_MODEL      e.g. "Qwen/Qwen2.5-72B-Instruct" or "llama3.3" or "openai/gpt-4o-mini"

If LLM_BASE_URL is unset, MockLLM is used automatically so the whole
pipeline is runnable and testable with zero infrastructure and zero
API keys -- this is what the eval harness and smoke tests use.
"""
from __future__ import annotations
import os
import json
import time
from typing import Any, Optional
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

CODE_VERSION = "2026-07-13-r12"  # bump on every fix so startup banner shows what's actually loaded

# Free-tier hosted models (OpenRouter free tier especially) hit transient
# rate limits under real traffic -- this is expected, not a bug in your
# setup. Retry with backoff rather than failing the whole request on the
# first 429/timeout/connection blip.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [5, 15, 30]

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def _call_with_retry(fn, *args, **kwargs):
    import openai
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_SECONDS[attempt]
                print(f"[llm] Transient error ({type(e).__name__}), retrying in {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
    raise last_exc


class LLMClient:
    """Thin wrapper so nodes never talk to the OpenAI SDK directly."""

    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL")
        self.api_key = os.getenv("LLM_API_KEY", "not-needed")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client = None
        if self.base_url:
            from openai import OpenAI  # openai SDK works against any OpenAI-compatible server
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def structured(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """
        Ask the model for output matching a Pydantic schema.

        Weaker/free models frequently drift from an implied schema (wrong
        field names, invented enum values, missing required fields) if you
        only describe it in prose. Two things fix this: (1) show the model
        the actual JSON schema, not just "respond in JSON", and (2) if it
        still fails validation, retry once with the exact error fed back --
        this alone fixes the large majority of first-try schema misses.
        """
        if self._client is None:
            return MockLLM.structured(system, user, schema)

        schema_json = schema.model_json_schema()
        full_system = (
            f"{system}\n\n"
            "Respond ONLY with a single valid JSON object -- no prose, no markdown fences, "
            "no explanation before or after. The object MUST match this exact JSON schema, "
            "using these exact field names and, where the schema gives an \"enum\", ONLY one "
            f"of the listed literal values (do not paraphrase or invent your own labels):\n\n"
            f"{json.dumps(schema_json, indent=2)}"
        )

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(2):  # first try, then one repair attempt
            resp = _call_with_retry(
                self._client.chat.completions.create,
                model=self.model, messages=messages, temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                parsed = json.loads(raw)
                return schema.model_validate(parsed)
            except Exception as e:  # JSONDecodeError or pydantic ValidationError
                last_error = e
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That response was invalid: {e}\n\n"
                        "Return ONLY a corrected JSON object matching the schema exactly -- "
                        "same field names, same enum values, nothing else."
                    ),
                })

        raise ValueError(
            f"Model failed to produce valid {schema.__name__} JSON after 2 attempts. "
            f"Last error: {last_error}. Last raw output: {raw!r}"
        )

    def complete(self, system: str, user: str) -> str:
        """Free-text completion, e.g. drafting a customer reply."""
        if self._client is None:
            return MockLLM.complete(system, user)

        resp = _call_with_retry(
            self._client.chat.completions.create,
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Offline mock -- deterministic, rule-based, zero dependencies.
# Lets you run/demo/test the ENTIRE graph with no model, no API key, no GPU.
# Swap it out by setting LLM_BASE_URL once you have a real endpoint.
# ---------------------------------------------------------------------------

class MockLLM:
    FRAUD_WORDS = {"stolen", "fraud", "hacked", "unauthorized", "scam", "chargeback"}
    ANGRY_WORDS = {"furious", "unacceptable", "lawsuit", "lawyer", "terrible", "worst", "scam"}

    @staticmethod
    def structured(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        from app.state import TriageResult, GuardrailVerdict  # local import to avoid cycles

        text = user.lower()

        if schema is TriageResult:
            if any(w in text for w in MockLLM.FRAUD_WORDS):
                category, urgency = "fraud_or_stolen", "high"
            elif "balance" in text:
                category, urgency = "balance_inquiry", "low"
            elif "redeem" in text or "code" in text or "pin" in text:
                category, urgency = "redemption_failed", "medium"
            elif "refund" in text or "charged twice" in text or "double charge" in text:
                category, urgency = "refund_request", "medium"
            elif "didn't arrive" in text or "not delivered" in text or "never got" in text or "never arrived" in text:
                category, urgency = "card_not_delivered", "medium"
            elif "login" in text or "password" in text or "locked out" in text:
                category, urgency = "account_access", "medium"
            else:
                category, urgency = "general_faq", "low"

            sentiment = "angry" if any(w in text for w in MockLLM.ANGRY_WORDS) else (
                "frustrated" if "!" in user or "again" in text else "neutral"
            )
            requires_money = category in {"refund_request", "fraud_or_stolen", "card_not_delivered"}

            return TriageResult(
                category=category,
                urgency=urgency,
                sentiment=sentiment,
                short_summary=f"Customer issue classified as {category} ({urgency} urgency).",
                requires_pii_or_money_action=requires_money,
            )

        if schema is GuardrailVerdict:
            # This branch is rarely hit -- guardrails.py implements the real
            # deterministic rules. This exists only so MockLLM satisfies the
            # LLMClient interface uniformly.
            return GuardrailVerdict(allow_auto_reply=True, reason="mock default", triggered_rules=[])

        raise NotImplementedError(f"MockLLM has no handler for schema {schema}")

    @staticmethod
    def complete(system: str, user: str) -> str:
        return (
            "Hi there, thanks for reaching out. Based on your account details, "
            "here is what I found and the next steps to resolve this. "
            "[MOCK REPLY -- set LLM_BASE_URL to generate real drafts]"
        )
