
#!/usr/bin/env python3
"""Full benchmark runner: GLM-5.3 x 9 tasks x 3 conditions (baseline, text_mas seq, text_mas hier).

Resumable: per-condition JSONL under results/; skip already-recorded (task, idx).
Concurrency: at most 16 benchmark API requests in flight.
Code tasks: candidate+gold executed ONLY in a throwaway docker sandbox
  (--network none --read-only --tmpfs --cap-drop ALL --no-new-privileges
   --user 65534 --memory 512m --pids-limit 64 --cpus 1, per-test timeout).
REPORT.md is regenerated after every completed condition.
"""
import asyncio, json, os, re, subprocess, sys, time, math
from typing import Dict, List, Optional

sys.path.insert(0, "/home/nvidia/latentmas-bench/harness")
import bench_lib as B
import prompts as lmas_prompts
from methods import default_agents

WS = "/home/nvidia/latentmas-bench"
RESULTS = f"{WS}/results"
LOGS = f"{WS}/logs"
MANIFESTS = f"{WS}/manifests"
MAX_IN_FLIGHT = 16

TASK_ORDER = ["aime2024", "aime2025", "gpqa", "humanevalplus", "mbppplus", "medqa",
              "gsm8k", "arc_easy", "arc_challenge"]
CONDITIONS = ["baseline", "text_mas_sequential", "text_mas_hierarchical"]

# ---------------- statistics ----------------
def wilson_ci(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0.0, center-half), min(1.0, center+half))

def mcnemar_exact(b, c):
    """Exact McNemar test on discordant counts b (base+ cond-), c (base- cond+)."""
    n = b + c
    if n == 0: return 1.0
    def binom_pmf_le(k, n, p=0.5):
        return sum(math.comb(n, i) * (0.5**n) for i in range(0, k+1))
    m = min(b, c)
    p = 2 * binom_pmf_le(m, n)
    return min(1.0, p)

def fmt(x, pct=False):
    if x is None: return "n/a"
    return f"{100*x:.1f}%" if pct else f"{x:.3f}"

# ---------------- code sandbox ----------------
SANDBOX = ["docker", "run", "--rm", "--network", "none", "--read-only",
           "--tmpfs", "/tmp:rw,size=64m", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--user", "65534:65534",
           "--memory", "512m", "--pids-limit", "64", "--cpus", "1",
           "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
           "-e", "MKL_NUM_THREADS=1", "-e", "NUMEXPR_NUM_THREADS=1",
           "-v"]

def sandbox_exec(code: str, timeout: int = 20) -> (bool, str):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(code)
        path = f.name
    os.chmod(path, 0o644)  # readable by container uid 65534
    try:
        cmd = SANDBOX + [f"{path}:/tmp/sol.py:ro", "latentmas-sandbox:py311",
                         "timeout", str(timeout), "python", "/tmp/sol.py"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+40)
        return (r.returncode == 0), (r.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        return False, "harness timeout"
    finally:
        os.unlink(path)

def extract_markdown_python_block(text):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None

# ---------------- scoring (mirrors repo exactly) ----------------
def score(task: str, final_text: str, gold: str, solution: str):
    if task in ["mbppplus", "humanevalplus"]:
        pred = extract_markdown_python_block(final_text)
        if pred is None:
            return None, False, "no python block"
        ok, err = sandbox_exec(pred + "\n" + gold)
        return pred, ok, err
    elif task in ["aime2024", "aime2025"]:
        pred = B.normalize_answer(B.extract_gsm8k_answer(final_text))
        try:
            ok = (int(pred) == int(str(gold).strip()))
        except (ValueError, TypeError):
            ok = False
        return pred, ok, None
    else:
        pred = B.normalize_answer(B.extract_gsm8k_answer(final_text))
        ok = (pred == gold) if (pred and gold) else False
        return pred, ok, None

# ---------------- conditions ----------------
async def run_item(client, sem, task, condition, idx, item) -> Dict:
    args = B.Args(task=task, method="text_mas" if condition != "baseline" else "baseline",
                  prompt="sequential" if condition == "text_mas_sequential" else "hierarchical")
    async with sem:
        t0 = time.time()
        if condition == "baseline":
            messages = lmas_prompts.build_agent_messages_single_agent(question=item["question"], args=args)
            outs = []
            out = await client.chat(messages)
            outs.append((messages, out))
            final_text = out["content"]
            traces = [{"name": "SingleAgent", "role": "singleagent",
                       "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
                       "reasoning_tokens": out["reasoning_tokens"], "latency_s": out["latency_s"],
                       "finish_reason": out["finish"]}]
        else:
            hierarchical = (condition == "text_mas_hierarchical")
            context, final_text = "", ""
            traces = []
            for agent in default_agents():
                if hierarchical:
                    messages = lmas_prompts.build_agent_messages_hierarchical_text_mas(
                        role=agent.role, question=item["question"], context=context, method="text_mas", args=args)
                else:
                    messages = lmas_prompts.build_agent_messages_sequential_text_mas(
                        role=agent.role, question=item["question"], context=context, method="text_mas", args=args)
                out = await client.chat(messages)
                text_out = out["content"].strip()
                if hierarchical:
                    name = B.AGENT_NAME_MAP_HIER[agent.name]
                else:
                    name = agent.name
                formatted = f"[{name}]:\n{text_out}\n\n"
                if agent.role != "judger":
                    context = f"{context}{formatted}"
                else:
                    final_text = text_out
                traces.append({"name": agent.name, "role": agent.role,
                               "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
                               "reasoning_tokens": out["reasoning_tokens"], "latency_s": out["latency_s"],
                               "finish_reason": out["finish"], "context_chars": len(context)})
        wall = time.time() - t0
    pred, ok, err = score(task, final_text, item.get("gold", ""), item.get("solution", ""))
    truncated = any(t["finish_reason"] == "length" for t in traces)
    return {
        "task": task, "condition": condition, "idx": idx,
        "question": item["question"], "gold": item.get("gold"), "prediction": pred,
        "raw_prediction": final_text, "correct": bool(ok), "exec_error": err,
        "truncated": truncated,
        "total_latency_s": round(wall, 4),
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "total_completion_tokens": sum(t["completion_tokens"] for t in traces),
        "total_reasoning_tokens": sum(t["reasoning_tokens"] for t in traces),
        "agents": traces,
    }

# ---------------- persistence ----------------
def tag_for(task, condition):
    return f"glm53_{task}_{condition.replace('text_mas_', 'textmas_')}"

def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "idx" in r:
                        done.add(r["idx"])
                except Exception:
                    pass
    return done

def append_result(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())

def load_records(path):
    if not os.path.exists(path): return []
    out = []
    with open(path) as f:
        for line in f:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

# ---------------- report ----------------
def build_report(summaries, base_dir=RESULTS):
    lines = ["# GLM-5.3 LatentMAS Benchmark Report (live)",
             "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
             "",
             "Conditions on GLM-5.3 (sglang, loopback 127.0.0.1:30001, TP=8, B300 x8):",
             "baseline (single agent), text_mas sequential (Planner->Critic->Refiner->Judger),",
             "text_mas hierarchical (Math/Science/Code -> Task Summarizer).",
             "LatentMAS latent_mas: NOT RUN on GLM-5.3 (needs in-process hidden-state/KV access).",
             "Prompts/scoring: official LatentMAS repo commit 9a9e4d3 (verbatim prompts.py, methods, scoring).",
             "Settings: temperature 0.6, top_p 0.95, max_tokens 4096, seed 42 sampling manifests.",
             "",
             "| Task | Condition | n | Acc | 95% CI | McNemar p vs base | prompt tok/item | completion tok/item | reasoning tok/item | mean lat (s) | wall (s) | truncation |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(summaries):
        s = summaries[key]
        lines.append(f"| {s['task']} | {s['condition']} | {s['n']} | {fmt(s['accuracy'], True)} "
                     f"| [{fmt(s['ci_lo'], True)}, {fmt(s['ci_hi'], True)}] | {fmt(s['mcnemar_p'])} "
                     f"| {s['prompt_per_item']:.0f} | {s['completion_per_item']:.0f} | {s['reasoning_per_item']:.0f} "
                     f"| {s['mean_latency_s']:.2f} | {s['wall_s']:.0f} | {s['truncation_rate']*100:.1f}% |")
    with open(f"{base_dir}/REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")

def make_summary(task, condition, records, wall_s, base_records):
    n = len(records)
    k = sum(1 for r in records if r.get("correct"))
    lo, hi = wilson_ci(k, n)
    lats = sorted(r["total_latency_s"] for r in records)
    if base_records:
        base_by_idx = {r["idx"]: bool(r.get("correct")) for r in base_records}
        b = c = 0
        for r in records:
            bv = base_by_idx.get(r["idx"])
            if bv is None: continue
            cv = bool(r.get("correct"))
            if bv and not cv: b += 1
            elif cv and not bv: c += 1
        p = mcnemar_exact(b, c)
    else:
        b = c = None; p = None
    return {
        "task": task, "condition": condition, "n": n, "correct": k,
        "accuracy": k/n if n else None, "ci_lo": lo, "ci_hi": hi,
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
        "wall_s": round(wall_s, 1),
        "mean_latency_s": round(sum(lats)/n, 3) if n else None,
        "median_latency_s": round(lats[n//2], 3) if n else None,
        "p95_latency_s": round(lats[min(n-1, int(0.95*n))], 3) if n else None,
        "prompt_per_item": sum(r.get("total_prompt_tokens", 0) for r in records)/max(1, n),
        "completion_per_item": sum(r.get("total_completion_tokens", 0) for r in records)/max(1, n),
        "reasoning_per_item": sum(r.get("total_reasoning_tokens", 0) or 0 for r in records)/max(1, n),
        "truncation_rate": sum(1 for r in records if r.get("truncated"))/max(1, n),
        "invalid_answer_rate": sum(1 for r in records if r.get("prediction") is None)/max(1, n),
        "tasks_per_s": round(n/wall_s, 4) if wall_s else None,
    }

async def main():
    only_tasks = sys.argv[1].split(",") if len(sys.argv) > 1 else TASK_ORDER
    sem = asyncio.Semaphore(MAX_IN_FLIGHT)
    client = B.GLMClient(max_tokens=4096)
    for task in TASK_ORDER:
        if task not in only_tasks: continue
        mpath = f"{MANIFESTS}/{task}_manifest.json"
        items = json.load(open(mpath))["items"]
        for condition in CONDITIONS:
            tag = tag_for(task, condition)
            path = f"{RESULTS}/{tag}.jsonl"
            done = load_done(path)
            todo = [(i, it) for i, it in enumerate(items) if i not in done]
            if not todo:
                log(f"[skip] {tag}: all {len(done)} done")
                continue
            log(f"[start] {tag}: {len(todo)} to run")
            t_start = time.time()
            CHUNK = 16
            for i0 in range(0, len(todo), CHUNK):
                chunk = todo[i0:i0+CHUNK]
                results = await asyncio.gather(*[
                    run_item(client, sem, task, condition, i, it) for i, it in chunk
                ], return_exceptions=True)
                for (i, it), res in zip(chunk, results):
                    if isinstance(res, Exception):
                        rec = {"task": task, "condition": condition, "idx": i, "question": it["question"],
                               "gold": it.get("gold"), "correct": False, "error": f"{type(res).__name__}: {res}",
                               "truncated": False, "total_latency_s": None,
                               "total_prompt_tokens": 0, "total_completion_tokens": 0, "total_reasoning_tokens": 0}
                        append_result(path, rec)
                    else:
                        append_result(path, res)
                log(f"[prog] {tag}: {min(i0+CHUNK, len(todo))}/{len(todo)}")
            wall = time.time() - t_start
            records = load_records(path)
            records = [r for r in records if r.get("condition") == condition and r.get("task") == task]
            base_path = f"{RESULTS}/{tag_for(task, 'baseline')}.jsonl"
            base_records = load_records(base_path) if condition != "baseline" else []
            s = make_summary(task, condition, records, wall, base_records)
            with open(f"{RESULTS}/{tag}_summary.json", "w") as f:
                json.dump(s, f, indent=2)
            log(f"[done] {tag}: acc={s['accuracy']} n={s['n']} wall={s['wall_s']}s")
            # regenerate REPORT.md from all existing summaries
            summaries = {}
            for fn in os.listdir(RESULTS):
                if fn.endswith("_summary.json"):
                    with open(f"{RESULTS}/{fn}") as f:
                        s2 = json.load(f)
                    summaries[s2["task"] + "|" + s2["condition"]] = s2
            build_report(summaries)
    await client.aclose()
    log("[all done]")

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f"{LOGS}/runner.log", "a") as f:
        f.write(line + "\n")

if __name__ == "__main__":
    asyncio.run(main())
