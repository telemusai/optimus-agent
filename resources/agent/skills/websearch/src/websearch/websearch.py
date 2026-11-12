"""Websearch skill implementation."""

from __future__ import annotations

import json
import os
import asyncio
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to default on bad values."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _agent_dir() -> Path:
    """Resolve the Optimus config dir the same way the runtime does."""
    raw = (
        os.environ.get("PRIME_AGENT_CODING_AGENT_DIR")
        or os.environ.get("PI_CODING_AGENT_DIR")
        or str(Path.home() / ".prime" / "agent")
    )
    return Path(raw).expanduser()


def _resolve_api_key() -> str:
    # Read auth.json on each call (not just the injected env var) so a key added
    # via /login after the kernel started is still picked up. Env var wins.
    env_key = os.environ.get("SERPER_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        auth = json.loads((_agent_dir() / "auth.json").read_text())
        cred = auth.get("serper") if isinstance(auth, dict) else None
        if isinstance(cred, dict) and cred.get("type") == "api_key":
            return _resolve_config_value(str(cred.get("key") or ""))
    except (OSError, ValueError):
        pass
    return ""


def _resolve_config_value(value: str) -> str:
    # Stored keys may be a literal or an env-var name; "!command" refs can't be run
    # safely here, so skip them (the agent injects those resolved at build time).
    value = value.strip()
    if not value or value.startswith("!"):
        return ""
    return (os.environ.get(value) or value).strip()


def _url_key(value: str) -> str:
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            return ""
        query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")])
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, ""))
    except ValueError:
        return ""


def _clean_text(value: object) -> str:
    return "".join(c for c in str(value or "") if c in "\n\t" or (ord(c) >= 32 and ord(c) != 127))


def _format_serper_results(data: dict, query: str, num_results: int = 5, seen: set[str] | None = None) -> str:
    """Format a Serper API response into readable text."""
    sections: list[str] = []

    kg = data.get("knowledgeGraph")
    if kg:
        kg_lines: list[str] = []
        title = (kg.get("title") or "").strip()
        if title:
            kg_lines.append(f"Knowledge Graph: {title}")
        description = (kg.get("description") or "").strip()
        if description:
            kg_lines.append(description)
        for key, value in (kg.get("attributes") or {}).items():
            text = str(value).strip()
            if text:
                kg_lines.append(f"{key}: {text}")
        if kg_lines:
            sections.append("\n".join(kg_lines))

    seen = set() if seen is None else seen
    for i, result in enumerate((data.get("organic") or [])[:num_results]):
        key = _url_key(str(result.get("link") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = key
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(snippet)
        sections.append("\n".join(lines))

    people_also_ask = data.get("peopleAlsoAsk") or []
    if people_also_ask:
        max_q = max(1, min(3, len(people_also_ask)))
        questions: list[str] = []
        for item in people_also_ask[:max_q]:
            question = (item.get("question") or "").strip()
            if not question:
                continue
            entry = f"Q: {question}"
            answer = (item.get("snippet") or "").strip()
            if answer:
                entry += f"\nA: {answer}"
            questions.append(entry)
        if questions:
            sections.append("People Also Ask:\n" + "\n".join(questions))

    if not sections:
        return f"No results returned for query: {query}"

    return "\n\n---\n\n".join(sections)


async def _fetch_serper(query: str, api_key: str, timeout: int = 45, num_results: int = 5) -> dict:
    """Execute a single Serper API search."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": num_results},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError("rate_limited" if e.response.status_code == 429 else "http_error") from None

    return data


def _native_binary() -> str | None:
    configured = os.environ.get("OPTIMUS_NATIVE_BINARY", "")
    if configured and Path(configured).is_file():
        return configured
    # Portable bundles can also run the skill CLI outside a live agent session.
    name = "optimus-rust.exe" if os.name == "nt" else "optimus-rust"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "bin" / name
        if candidate.is_file():
            return str(candidate)
    return None


async def _fetch_codex(query: str, timeout: int, num_results: int, mode: str, model: str) -> dict:
    binary = _native_binary()
    if binary is None:
        return {"status": "unavailable", "error": "native_search_unavailable"}
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            binary, "--internal-codex-web-search", stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, **kwargs,
        )
        request = json.dumps({"query": query, "timeout_seconds": timeout, "num_results": num_results,
                              "mode": mode, "model": model}).encode()
        output, _ = await asyncio.wait_for(process.communicate(request), timeout + 3)
        if process.returncode != 0 or len(output) > 262144:
            return {"status": "error", "error": "native_search_failed"}
        response = json.loads(output)
        if not isinstance(response, dict) or response.get("status") not in ("ok", "unavailable", "error"):
            return {"status": "error", "error": "invalid_response"}
        if response["status"] == "ok" and (
            not isinstance(response.get("summary"), str)
            or not isinstance(response.get("sources"), list)
            or not all(isinstance(source, dict) for source in response["sources"])
        ):
            return {"status": "error", "error": "invalid_response"}
        return response
    except asyncio.TimeoutError:
        return {"status": "error", "error": "timeout"}
    except (OSError, ValueError):
        return {"status": "error", "error": "native_search_failed"}
    finally:
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


def _format_codex_results(data: dict, mode: str, seen: set[str], budget: int) -> str:
    summary = re.sub(r"\ue200[^\ue201]*\ue201", "", _clean_text(data.get("summary", "")))
    lines = [f"Codex web search ({mode})", _truncate(summary, min(4000, budget // 2))]
    for source in data.get("sources", []):
        url = _url_key(str(source.get("url") or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        title = _clean_text(source.get("title") or url).replace("\n", " ").replace("[", "").replace("]", "")
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines)


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    marker = "\n... [output truncated] ...\n"
    if budget <= len(marker):
        return text[:budget]
    left = (budget - len(marker)) // 2
    right = budget - len(marker) - left
    return text[:left] + marker + text[-right:]


async def run(
    query: str,
    *,
    max_output: int = 8192,
    timeout: int | None = None,
    num_results: int | None = None,
    provider: str = "auto",
    mode: str = "live",
    codex_model: str = "gpt-6-astra",
) -> str:
    """Search with authenticated Codex and append available Serper results.

    Args:
        query: Google search query.
        max_output: Truncate output to this many chars.
        timeout: HTTP timeout in seconds.
        num_results: Organic results to return.
        provider: auto combines configured providers; codex or serper selects one.
        mode: live or cached Codex search (Serper always uses live Google search).
        codex_model: Codex model used for hosted search, independent of the chat model.

    Returns:
        Formatted search results.
    """
    if not isinstance(query, str) or not query.strip() or len(query.encode()) > 8192:
        raise ValueError("query must contain 1–8192 UTF-8 bytes")
    if provider not in ("auto", "codex", "serper") or mode not in ("live", "cached"):
        raise ValueError("provider must be auto/codex/serper and mode must be live/cached")
    if not 1 <= max_output <= 131072:
        raise ValueError("max_output must be between 1 and 131072")
    if timeout is None:
        timeout = _env_int("PRIME_AGENT_WEBSEARCH_TIMEOUT", 45)
    if num_results is None:
        num_results = _env_int("PRIME_AGENT_WEBSEARCH_NUM_RESULTS", 5)
    timeout = max(1, min(90, timeout))
    num_results = max(1, min(20, num_results))
    api_key = _resolve_api_key() if provider != "codex" else ""
    names, tasks = [], []
    if provider != "serper":
        names.append("codex")
        tasks.append(_fetch_codex(query, timeout, num_results, mode, codex_model))
    if api_key:
        names.append("serper")
        tasks.append(_fetch_serper(query, api_key, timeout, num_results))
    replies = dict(zip(names, await asyncio.gather(*tasks, return_exceptions=True)))
    sections, notices, seen = [], [], set()
    codex = replies.get("codex")
    if isinstance(codex, dict) and codex.get("status") == "ok":
        sections.append(_format_codex_results(codex, mode, seen, max_output))
    elif isinstance(codex, Exception) or isinstance(codex, dict) and codex.get("status") == "error":
        notices.append("Codex search unavailable; check the existing Codex login or credential configuration.")
    serper = replies.get("serper")
    if isinstance(serper, dict):
        try:
            sections.append("Serper Google search\n" + _clean_text(_format_serper_results(serper, query, num_results, seen)))
        except (TypeError, ValueError, AttributeError):
            serper = ValueError("invalid_response")
    if isinstance(serper, Exception):
        notices.append("Serper search unavailable; retry or check Serper in /login → MCP Connections.")
    if not sections:
        return _truncate("\n".join(notices + [
            "No web search results. Check the existing OpenAI Codex login or credential configuration; "
            "if Codex is not configured in Optimus Agent, use /login to sign in, "
            "or configure Serper (web search) under MCP Connections. "
            "Start an updated Optimus session if the native search helper is unavailable."
        ]), max_output)
    prefix = f'Results for query "{_clean_text(query)}":\n\n'
    suffix = ("\n\n" + "\n".join(notices)) if notices else ""
    budget = max(0, (max_output - len(prefix) - len(suffix) - 7 * (len(sections) - 1)) // len(sections))
    return _truncate(prefix + "\n\n---\n\n".join(_truncate(s, budget) for s in sections) + suffix, max_output)
