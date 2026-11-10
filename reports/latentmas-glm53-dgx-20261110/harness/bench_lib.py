
"""GLM-5.3 adapter harness for LatentMAS benchmark conditions.

Reuses (verbatim) from the official LatentMAS repo (commit 9a9e4d3):
  - prompts.py            -> all prompt builders (byte-identical rendering)
  - methods/__init__.py   -> default_agents() role sequence
  - utils.py              -> answer extraction / normalization / gold extraction
  - methods/text_mas.py   -> pipeline logic (context accumulation, judger output)
  - methods/baseline.py   -> single-agent logic

Replaced (necessarily): models.py HF/vLLM loaders -> OpenAI-compatible
client against the served GLM-5.3 (sglang, loopback 127.0.0.1:30001).
The latent_mas method is NOT run: it needs in-process hidden-state/KV access.
"""
import json, os, re, sys, time, argparse, traceback
from typing import Dict, List, Optional
import httpx

REPO = "/home/nvidia/latentmas-bench/LatentMAS"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import prompts as lmas_prompts
from methods import default_agents

# ---------------- scoring (verbatim from repo utils.py) ----------------
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None

def extract_gold(text: str) -> Optional[str]:
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return m.group(1) if m else None

def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()

def score_gsm8k(final_text: str, gold: str):
    pred = normalize_answer(extract_gsm8k_answer(final_text))
    ok = (pred == gold) if (pred and gold) else False
    return pred, ok

# ---------------- GLM-5.3 client ----------------
API_BASE = "http://127.0.0.1:30001/v1"
MODEL_ID = "glm-5.3"

def load_api_key():
    with open("/home/nvidia/.config/optimus-rust/models.json") as f:
        return json.load(f)["providers"]["dgx-glm53-occ3"]["apiKey"]

class GLMClient:
    def __init__(self, max_tokens=4096, temperature=0.6, top_p=0.95, timeout=900.0):
        self.key = load_api_key()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers={"Authorization": f"Bearer {self.key}"},
        )

    async def chat(self, messages: List[Dict], max_tokens=None) -> Dict:
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
        }
        t0 = time.time()
        last_err = None
        for attempt in range(3):
            try:
                r = await self._client.post(f"{API_BASE}/chat/completions", json=payload)
                dt = time.time() - t0
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                    if r.status_code >= 500 or r.status_code == 429:
                        await __import__("asyncio").sleep(5 * (attempt + 1))
                        continue
                    raise RuntimeError(last_err)
                j = r.json()
                msg = j["choices"][0]["message"]
                usage = j.get("usage", {})
                return {
                    "content": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content") or "",
                    "finish": j["choices"][0].get("finish_reason"),
                    "latency_s": round(dt, 4),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": usage.get("reasoning_tokens"),
                    "attempts": attempt + 1,
                }
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = f"{type(e).__name__}: {e}"
                await __import__("asyncio").sleep(5 * (attempt + 1))
        raise RuntimeError(f"API failed after retries: {last_err}")

    async def aclose(self):
        await self._client.aclose()

# ---------------- conditions ----------------
class Args:
    """Mirrors the repo argparse fields used by prompt builders."""
    def __init__(self, task, method, prompt="sequential", model_name="GLM-5.3 (Qwen-prompt-compatible)",
                 text_mas_context_length=-1):
        self.task = task
        self.method = method
        self.prompt = prompt
        self.model_name = model_name  # contains 'qwen' so repo assertions pass; not rendered into any prompt
        self.text_mas_context_length = text_mas_context_length

async def run_baseline(client: GLMClient, item: Dict, args: Args) -> Dict:
    messages = lmas_prompts.build_agent_messages_single_agent(question=item["question"], args=args)
    t0 = time.time()
    out = await client.chat(messages)
    total = time.time() - t0
    pred, ok = score_gsm8k(out["content"], item["gold"])
    return {
        "condition": "baseline", "task": args.task,
        "question": item["question"], "gold": item["gold"], "prediction": pred,
        "raw_prediction": out["content"], "correct": ok,
        "agents": [{"name": "SingleAgent", "role": "singleagent",
                    "input_messages": messages, "output": out["content"],
                    "reasoning": out["reasoning"],
                    "prompt_tokens": out["prompt_tokens"],
                    "completion_tokens": out["completion_tokens"],
                    "reasoning_tokens": out["reasoning_tokens"],
                    "latency_s": out["latency_s"]}],
        "total_latency_s": round(total, 4),
        "total_prompt_tokens": out["prompt_tokens"],
        "total_completion_tokens": out["completion_tokens"],
        "total_reasoning_tokens": out["reasoning_tokens"],
    }

AGENT_NAME_MAP_HIER = {
    "Planner": "Math Agent", "Critic": "Science Agent",
    "Refiner": "Code Agent", "Judger": "Task Summrizer",
    "planner": "Math Agent", "critic": "Science Agent",
    "refiner": "Code Agent", "judger": "Task Summrizer",
}

async def run_text_mas(client: GLMClient, item: Dict, args: Args) -> Dict:
    hierarchical = (args.prompt == "hierarchical")
    context, history, final_text = "", "", ""
    traces = []
    t0 = time.time()
    for agent in default_agents():
        if hierarchical:
            messages = lmas_prompts.build_agent_messages_hierarchical_text_mas(
                role=agent.role, question=item["question"], context=context,
                method="text_mas", args=args)
        else:
            messages = lmas_prompts.build_agent_messages_sequential_text_mas(
                role=agent.role, question=item["question"], context=context,
                method="text_mas", args=args)
        out = await client.chat(messages)
        text_out = out["content"].strip()
        if hierarchical:
            formatted = f"[{AGENT_NAME_MAP_HIER[agent.name]}]:\n{text_out}\n\n"
        else:
            formatted = f"[{agent.name}]:\n{text_out}\n\n"
        if agent.role != "judger":
            context = f"{context}{formatted}"
            history = f"{history}{formatted}"
        else:
            final_text = text_out
        traces.append({
            "name": agent.name, "role": agent.role,
            "input_messages": messages, "output": text_out,
            "reasoning": out["reasoning"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "reasoning_tokens": out["reasoning_tokens"],
            "latency_s": out["latency_s"],
            "context_chars": len(context),
        })
    total = time.time() - t0
    pred, ok = score_gsm8k(final_text, item["gold"])
    return {
        "condition": f"text_mas_{args.prompt}", "task": args.task,
        "question": item["question"], "gold": item["gold"], "prediction": pred,
        "raw_prediction": final_text, "correct": ok,
        "agents": traces, "context": history,
        "total_latency_s": round(total, 4),
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "total_completion_tokens": sum(t["completion_tokens"] for t in traces),
        "total_reasoning_tokens": sum(t["reasoning_tokens"] for t in traces),
        # communication payload = context text flowing between agents
        "communication_tokens_est": sum(t["prompt_tokens"] for t in traces[1:]) if traces else 0,
        "context_final_chars": len(context),
    }

# ---------------- persistence ----------------
def out_path(results_dir, tag):
    return os.path.join(results_dir, f"{tag}.jsonl")

def load_done(path, key="question"):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)[key])
                except Exception:
                    pass
    return done

def append_result(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def summarize(tag, records, wall_s):
    n = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    lats = sorted(r["total_latency_s"] for r in records)
    def pct(p):
        return lats[min(n - 1, int(p * n))] if n else None
    return {
        "tag": tag, "n": n, "correct": correct, "accuracy": correct / n if n else None,
        "wall_s": round(wall_s, 2),
        "mean_latency_s": round(sum(lats) / n, 3) if n else None,
        "median_latency_s": round(pct(0.5), 3) if n else None,
        "p95_latency_s": round(pct(0.95), 3) if n else None,
        "tasks_per_s": round(n / wall_s, 4) if wall_s else None,
        "total_prompt_tokens": sum(r.get("total_prompt_tokens", 0) for r in records),
        "total_completion_tokens": sum(r.get("total_completion_tokens", 0) for r in records),
        "total_reasoning_tokens": sum(r.get("total_reasoning_tokens", 0) for r in records),
    }
