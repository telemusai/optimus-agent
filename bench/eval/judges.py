"""DGX-only judge adapters with explicit source and protocol boundaries.

LongMemEval templates are copied from the MIT-licensed get_anscheck_prompt
at 9e0b455f4ef0e2ab8f2e582289761153549043fc. Copyright (c) 2024 Di Wu.
LME-V2 prompts/parsers are modified Apache-2.0 source at
2cc8c540bdb87fe6761629b585e727e1c4704520. See ../THIRD_PARTY_NOTICES.md
and ../licenses/ for complete notices and function-level provenance.
Modified: separate adapters, explicit DGX config, offline default, stdlib
transport, secret-safe diagnostics, request cache, and campaign budgets.

LoCoMo-Plus upstream prompts/parsers are not distributed. Its local rubric
comes from the independent local_scoring.py implementation and is a new metric.
No installed-profile lookup, credential command, netrc, proxy, or redirect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from bench.protocol import EVALUATOR_VERSION
from .local_scoring import make_judge_request, parse_judge_response, SCORING_VERSION

DEFAULT_PROVIDER = "dgx-glm53"
DEFAULT_MODEL_ID = "glm-5.3"

# ---------------------------------------------------------------------------
# MIT LongMemEval template constants (source pin in THIRD_PARTY_NOTICES.md)
# ---------------------------------------------------------------------------

LME_TEMPLATE_SINGLE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate steps "
    "to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
LME_TEMPLATE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate steps "
    "to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the answer, answer no. "
    "In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the model makes "
    "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response "
    "is still correct. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
LME_TEMPLATE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the response "
    "should be considered as correct as long as the updated answer is the required answer.\n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
LME_TEMPLATE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
    "answer no. The model does not need to reflect all the points in the rubric. The response is "
    "correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
LME_TEMPLATE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is given "
    "but the asked information is not.\n\n"
    "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)

LME_TEMPLATES = {
    "single-session-user": LME_TEMPLATE_SINGLE,
    "single-session-assistant": LME_TEMPLATE_SINGLE,
    "multi-session": LME_TEMPLATE_SINGLE,
    "temporal-reasoning": LME_TEMPLATE_TEMPORAL,
    "knowledge-update": LME_TEMPLATE_KNOWLEDGE_UPDATE,
    "single-session-preference": LME_TEMPLATE_PREFERENCE,
}


def build_longmemeval_prompt(task: str, question, answer, response, abstention: bool = False) -> str:
    """Build a message from the retained MIT template constants."""
    if not abstention:
        template = LME_TEMPLATES[task]
    else:
        template = LME_TEMPLATE_ABSTENTION
    return template.format(str(question), str(answer), str(response))


# ---------------------------------------------------------------------------
# Apache-2.0 LME-V2 prompt/parser adaptations (pinned in notices)
# ---------------------------------------------------------------------------

V2_ABSTENTION_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for flawed-premise (abstention) questions. "
    "Judge whether a model answer correctly identifies that the question premise is wrong, "
    "consistent with the reference answer. "
    "If the model follows the flawed premise and gives a concrete answer under that premise, "
    "it must be graded 0. "
    "If the model's final answer is just UNKNOWN / cannot determine without identifying the flaw, grade 0. "
    "If the model is contradictory (both rejects premise and also gives a concrete premise-following answer), grade 0. "
    "Paraphrases are allowed when they preserve the same core flaw described by the reference answer."
)

V2_GOTCHAS_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for gotchas-style insight questions. "
    "The reference answer describes the key insight(s). "
    "Grade 1 if the model response includes at least one correct insight point from the reference answer "
    "(paraphrase allowed), and does not contradict any reference point. "
    "If the model's direction is wrong, or it contains contradictions against any reference point, grade 0. "
    "If the model gives multiple points, partial coverage is enough for 1 as long as no contradictions appear."
)


def v2_build_abstention_judge_messages(
    *,
    question_text: str,
    reference_answer: str,
    model_full_response: str,
    model_final_answer: str,
) -> List[Dict[str, str]]:
    """Licensed LME-V2 abstention message builder adaptation."""
    user_prompt = (
        "Evaluate whether the model correctly handled a flawed-premise question.\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Reference answer (the expected flaw-aware conclusion):\n"
        f"{reference_answer}\n\n"
        "Model full response:\n"
        f"{model_full_response}\n\n"
        "Model extracted final answer:\n"
        f"{model_final_answer}\n\n"
        "Scoring rubric:\n"
        "- Label 1 only if the model clearly identifies the flawed premise and reaches a conclusion "
        "equivalent to the reference answer.\n"
        "- Label 1 also if the model explicitly says it lacks access to the user's specific "
        "live environment/instance/configuration and therefore cannot verify, provided it does not "
        "give a concrete premise-following answer.\n"
        "- Label 0 if the model follows the flawed premise and gives a concrete answer under that premise.\n"
        "- Label 0 for generic UNKNOWN/insufficient-info replies that do not identify a flaw and do not "
        "make the explicit environment-access limitation clear.\n"
        "- Label 0 if contradictory.\n\n"
        "Output JSON only:\n"
        '{"label": 0 or 1, "reason": "short rationale"}'
    )
    return [
        {"role": "system", "content": V2_ABSTENTION_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def v2_build_gotchas_judge_messages(
    *,
    question_text: str,
    reference_answer: str,
    model_full_response: str,
    model_final_answer: str,
) -> List[Dict[str, str]]:
    """Licensed LME-V2 gotchas message builder adaptation."""
    user_prompt = (
        "Evaluate whether the model answer captures the gotcha insight.\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Reference answer (insight points):\n"
        f"{reference_answer}\n\n"
        "Model full response:\n"
        f"{model_full_response}\n\n"
        "Model extracted final answer:\n"
        f"{model_final_answer}\n\n"
        "Scoring rubric:\n"
        "- Label 1 if the model includes at least one correct insight point from the reference answer "
        "(paraphrase acceptable), and does not contradict any reference point.\n"
        "- Label 1 even if only part of a multi-point reference answer is covered, as long as there is "
        "no contradiction.\n"
        "- Label 0 if direction is wrong (suggests opposite action/cause), even if some wording overlaps.\n"
        "- Label 0 if any point in the model response contradicts any reference point.\n"
        "- Label 0 if the response is irrelevant or generic without insight.\n\n"
        "Output JSON only:\n"
        '{"label": 0 or 1, "reason": "short rationale"}'
    )
    return [
        {"role": "system", "content": V2_GOTCHAS_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _v2_stringify_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _v2_strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def v2_parse_llm_binary_judgement(text: str) -> Tuple[int, str]:
    """Licensed LME-V2 label parser adaptation; rejects unparseable text."""
    cleaned = _v2_strip_markdown_code_fence(_v2_stringify_text(text))
    if not cleaned:
        raise ValueError("Empty judgement response from evaluator model.")

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        json_blob = json_match.group(0)
        try:
            payload = json.loads(json_blob)
            if not isinstance(payload, dict):
                raise ValueError("Evaluator JSON payload must be an object.")
            label = payload.get("label")
            if label in {0, 1, "0", "1"}:
                label_int = int(label)
                reason = _v2_stringify_text(payload.get("reason"))
                return label_int, reason
        except json.JSONDecodeError:
            pass

    label_match = re.search(r'"label"\s*:\s*([01])', cleaned, flags=re.IGNORECASE)
    if not label_match:
        label_match = re.search(r"'label'\s*:\s*([01])", cleaned, flags=re.IGNORECASE)
    if not label_match:
        label_match = re.search(r"\blabel\b\s*[:=]\s*([01])", cleaned, flags=re.IGNORECASE)
    if label_match:
        return int(label_match.group(1)), cleaned

    raise ValueError(f"Could not parse evaluator binary judgement: {cleaned!r}")


# ---------------------------------------------------------------------------
# DGX endpoint client
# ---------------------------------------------------------------------------


def _resolve_secret(value) -> str:
    """Resolve a literal or {"env": "NAME"}; never execute credential commands."""
    if isinstance(value, dict) and set(value) == {"env"}:
        name = value["env"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("invalid credential environment variable name")
        value = os.environ.get(name)
        if not value:
            raise ValueError("configured credential environment variable is empty")
    if not isinstance(value, str):
        raise ValueError("credentials must be literals or explicit env references")
    if value.lstrip().startswith("!"):
        raise ValueError("credential commands are not supported; use an explicit env reference")
    if "\r" in value or "\n" in value:
        raise ValueError("credential values must be single-line")
    return value


class DGXClient:
    """Explicit DGX GLM-5.3 client. Cache-only unless network is opted in.

    No installed profile, shell resolver, proxy, or netrc lookup is used.
    An injected transport receives (url, headers, body, timeout) and returns
    (status_code, parsed_json). It is intended for offline fixtures.
    """

    def __init__(
        self,
        models_json: Optional[Path] = None,
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL_ID,
        transport: Optional[Callable[..., Tuple[int, Any]]] = None,
        timeout_s: float = 300.0,
        retries: int = 3,
        cache_dir: Optional[Path] = None,
        allow_network: bool = False,
    ):
        if provider != DEFAULT_PROVIDER or model != DEFAULT_MODEL_ID:
            raise ValueError("benchmark judges require DGX provider dgx-glm53 and model glm-5.3")
        if retries < 1 or timeout_s <= 0:
            raise ValueError("judge retries and timeout must be positive")
        config_path = models_json or os.environ.get("MEMORY_BENCH_MODELS_JSON")
        self.models_json = Path(config_path) if config_path else None
        self.model = model
        self.base_url = ""
        self.api_key = ""
        self.extra_headers = {}
        self._secrets = []
        if self.models_json is not None:
            self._load_config()
        if allow_network and not self.base_url:
            raise ValueError("live judging requires an explicit --models-json configuration")
        self.timeout_s = timeout_s
        self.retries = retries
        cache_root = cache_dir or os.environ.get("MEMORY_BENCH_JUDGE_CACHE")
        self.cache_dir = Path(cache_root) if cache_root else None
        self.allow_network = allow_network
        self._transport = transport or self._default_transport
        self._injected_transport = transport is not None
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0, "failures": 0}

    def _load_config(self):
        from urllib.parse import urlsplit

        try:
            cfg = json.loads(self.models_json.read_text(encoding="utf-8"))
            prov = cfg["providers"][DEFAULT_PROVIDER]
            url = prov["baseUrl"]
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("invalid endpoint")
            self.base_url = url.rstrip("/")
            self.api_key = _resolve_secret(prov["apiKey"])
            headers = prov.get("headers") or {}
            for key, value in headers.items():
                if (not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key)
                        or key.lower() in {"authorization", "host", "content-length"}):
                    raise ValueError("invalid header")
                self.extra_headers[key] = _resolve_secret(value)
            self._secrets = [v for v in [self.api_key, *self.extra_headers.values()] if v]
        except (KeyError, TypeError, ValueError, OSError):
            raise ValueError("invalid explicit DGX judge configuration (details omitted)") from None

    def redact(self, value):
        """Do not persist configured credentials echoed by an endpoint."""
        if isinstance(value, str):
            for secret in sorted(self._secrets, key=len, reverse=True):
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self.redact(k): self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        return value

    def _default_transport(self, url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float):
        # Importing this module never creates a client or opens a connection.
        from urllib.error import HTTPError, URLError
        from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                return None

        request = Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        opener = build_opener(ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            status = exc.code
            exc.close()
            return status, {}  # response bodies may contain credentials
        except (URLError, OSError, ValueError):
            raise RuntimeError("judge transport failed (details omitted)") from None

    def _post_chat(self, messages: List[Dict[str, str]], *, temperature=None, max_tokens: int = 2048) -> Tuple[str, Dict[str, Any]]:
        if not self.allow_network and not self._injected_transport:
            raise RuntimeError("judge cache miss; network disabled (explicit --allow-network required)")
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            body["temperature"] = temperature
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.extra_headers,
        }
        url = f"{self.base_url}/chat/completions"
        last_err = None
        for attempt in range(self.retries):
            try:
                status, payload = self._transport(url, headers, body, self.timeout_s)
                if status == 200:
                    try:
                        content = payload["choices"][0]["message"]["content"] or ""
                    except (KeyError, IndexError, TypeError):
                        content = ""
                    if not str(content).strip():
                        last_err = RuntimeError("empty content in judge response")
                    else:
                        return str(content), {
                            "status": status,
                            "usage": payload.get("usage") or {},
                            "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
                        }
                else:
                    last_err = RuntimeError(f"HTTP {status} from judge endpoint")
            except Exception:
                last_err = RuntimeError("judge transport failed (details omitted)")
            self.stats["retries"] += 1
            if attempt + 1 < self.retries:
                time.sleep(0.5 * (2 ** attempt))
        self.stats["failures"] += 1
        raise last_err if last_err else RuntimeError("judge call failed")

    def chat_cached(
        self,
        judge_name: str,
        messages: List[Dict[str, str]],
        *,
        temperature=None,
        max_tokens: int = 2048,
    ) -> Tuple[str, Dict[str, Any]]:
        # New protocol namespace intentionally cannot reuse historical campaign caches.
        cache_key = {
            "protocol": EVALUATOR_VERSION,
            "judge": judge_name,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        digest = hashlib.sha256(
            json.dumps(cache_key, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{digest}.json" if self.cache_dir else None
        if cache_path is not None and cache_path.is_file():
            try:
                rec = json.loads(cache_path.read_text(encoding="utf-8"))
                if (rec.get("protocol") != EVALUATOR_VERSION or rec.get("request_sha256") != digest
                        or rec.get("model") != self.model):
                    raise ValueError("cache protocol/request identity mismatch")
                content = rec["content"]
                if not isinstance(content, str):
                    raise ValueError("invalid cache content")
                self.stats["cache_hits"] += 1
                return self.redact(content), self.redact(rec.get("meta", {}))
            except (ValueError, KeyError, TypeError):
                pass  # offline cache misses never fall through to a live call

        t0 = time.time()
        content, meta = self._post_chat(messages, temperature=temperature, max_tokens=max_tokens)
        latency_ms = int((time.time() - t0) * 1000)
        content, meta = self.redact(content), self.redact(meta)
        meta = dict(meta, request_sha256=digest, protocol=EVALUATOR_VERSION)
        record = {
            "protocol": EVALUATOR_VERSION,
            "request_sha256": digest,
            "judge": judge_name,
            "model": self.model,
            "params": {"temperature": temperature, "max_tokens": max_tokens},
            "prompt_sha256": hashlib.sha256(
                json.dumps(messages, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "content": content,
            "meta": dict(meta, latency_ms=latency_ms),
            "latency_ms": latency_ms,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cached": False,
        }
        if cache_path is not None:
            # Unique temp names permit concurrent identical requests without races.
            import tempfile
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_dir,
                                                 prefix=digest + ".", suffix=".tmp", delete=False) as handle:
                    tmp = Path(handle.name)
                    json.dump(record, handle, ensure_ascii=False, indent=1)
                os.replace(tmp, cache_path)
            finally:
                if tmp is not None and tmp.exists():
                    tmp.unlink()
        self.stats["requests"] += 1
        return content, dict(meta, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Judge entry points
# ---------------------------------------------------------------------------

_shared_client: Optional[DGXClient] = None


def get_client(**kwargs) -> DGXClient:
    global _shared_client
    if _shared_client is None or kwargs:
        _shared_client = DGXClient(**kwargs)
    return _shared_client


def set_client(client: DGXClient) -> None:
    """Inject a client (used by selftest / tests)."""
    global _shared_client
    _shared_client = client


def judge_longmemeval(
    question, answer, response, prompt_variant: str, abstention: bool = False
) -> Dict[str, Any]:
    """MIT-template LongMemEval adapter. Returns {"score": 0|1, "label": bool, ...}.

    Official params: temperature 0, max_tokens 10 -> RAISED to 2048 for the
    reasoning glm-5.3 (documented deviation; prompt unchanged).
    """
    prompt = build_longmemeval_prompt(prompt_variant, question, answer, response, abstention)
    client = get_client()
    content, meta = client.chat_cached(
        "longmemeval",
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2048,
    )
    label = "yes" in content.lower()  # official: eval_response.strip() then substring
    return {
        "score": 1.0 if label else 0.0,
        "label": label,
        "judge_raw": content.strip(),
        "judge_meta": meta,
        "judge": "longmemeval",
        "prompt_variant": prompt_variant,
        "abstention": abstention,
    }


def judge_locomo_plus(evidence_cue: str, prediction: str, *, question: str = "",
                       max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """New local semantic metric, not LoCoMo-Plus upstream Cognitive scoring."""
    request = make_judge_request(question=question, reference_answer=evidence_cue,
                                 candidate_answer=prediction)
    tokens = int(os.environ.get("LP_JUDGE_MAX_TOKENS", "512")) if max_tokens is None else max_tokens
    if tokens < 1:
        raise ValueError("local judge token budget must be positive")
    content, meta = get_client().chat_cached(
        "locomo_plus/" + SCORING_VERSION, request.as_messages(), temperature=0, max_tokens=tokens,
    )
    result = parse_judge_response(content)
    return {"score": float(result.score), "label": result.correct,
            "judge_reason": result.reason, "judge_raw": content.strip(), "judge_meta": meta,
            "judge": "locomo_plus_local", "judge_protocol": SCORING_VERSION,
            "judge_request_identity": request.cache_identity}


def judge_lme_v2_abstention(
    question, reference_answer, full_response, final_answer
) -> Dict[str, Any]:
    """Licensed LME-V2 abstention adapter. Upstream parameters:
    temperature unset (omitted), max_completion_tokens 4096 (sent as max_tokens)."""
    messages = v2_build_abstention_judge_messages(
        question_text=_v2_stringify_text(question),
        reference_answer=_v2_stringify_text(reference_answer),
        model_full_response=_v2_stringify_text(full_response),
        model_final_answer=_v2_stringify_text(final_answer),
    )
    client = get_client()
    content, meta = client.chat_cached(
        "lme_v2_abstention", messages, temperature=None, max_tokens=4096
    )
    label_int, reason = v2_parse_llm_binary_judgement(content)
    return {
        "score": 1.0 if label_int == 1 else 0.0,
        "label": label_int,
        "judge_reason": reason,
        "judge_raw": content.strip(),
        "judge_meta": meta,
        "judge": "lme_v2_abstention",
    }


def judge_lme_v2_gotchas(
    question, reference_answer, full_response, final_answer
) -> Dict[str, Any]:
    """Licensed LME-V2 gotchas adapter. Parameters as abstention adapter."""
    messages = v2_build_gotchas_judge_messages(
        question_text=_v2_stringify_text(question),
        reference_answer=_v2_stringify_text(reference_answer),
        model_full_response=_v2_stringify_text(full_response),
        model_final_answer=_v2_stringify_text(final_answer),
    )
    client = get_client()
    content, meta = client.chat_cached(
        "lme_v2_gotchas", messages, temperature=None, max_tokens=4096
    )
    label_int, reason = v2_parse_llm_binary_judgement(content)
    return {
        "score": 1.0 if label_int == 1 else 0.0,
        "label": label_int,
        "judge_reason": reason,
        "judge_raw": content.strip(),
        "judge_meta": meta,
        "judge": "lme_v2_gotchas",
    }
