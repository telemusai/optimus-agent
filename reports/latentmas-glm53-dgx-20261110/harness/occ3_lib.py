
"""OCC3 condition orchestrator: 4-agent sequential chain with native occ3_handoff between agents.

Design (operator directive 2026-11-10):
- Spawn order: judger, refiner, critic, planner (receivers before senders).
- Each sender: do role work -> occ3_handoff (arm) -> emit concise output text -> occ3_handoff (publish).
- Each receiver: standby -> native state injected -> role work -> next handoff (or final answer).
- Item counts as OCC3 (native) only if server-side capture/send counters moved AND sender reported
  ADMITTED/delivered. Fallbacks to ordinary messaging are recorded as fallback items.
- Judger must end with 'FINAL_ANSWER: <ans>' for scoring (same extract/normalize as repo).
"""
import asyncio, json, os, re, sys, time
import httpx

WS = "/home/nvidia/latentmas-bench"
RESULTS = f"{WS}/results"
OCC3_OUT = f"{RESULTS}/glm53_gsm8k_occ3.jsonl"
MODEL = "dgx-glm53-occ3/glm-5.3"
ITEM_TIMEOUT_S = 600  # per-item cap

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)

def load_api_key():
    with open("/home/nvidia/.config/optimus-rust/models.json") as f:
        return json.load(f)["providers"]["dgx-glm53-occ3"]["apiKey"]

async def occ3_counters():
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get("http://127.0.0.1:30001/occ3/v1/status",
                            headers={"Authorization": f"Bearer {load_api_key()}"})
        m = r.json()["metrics"]["counters"]
        return {k: m[k] for k in ("send_ok", "send_fail", "capture_ok", "capture_fail")}
    except Exception as e:
        return {"error": str(e)}

# repo scoring (verbatim logic)
def extract_gsm8k_answer(text):
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None

def normalize_answer(ans):
    return None if ans is None else ans.strip().lower()

WAIT_RULE = (
    "CRITICAL WAIT RULE: while waiting for the native state, end your turn WITHOUT any RLM_CHILD_STATUS line "
    "(the harness will continue you automatically when new turns arrive). Never emit 'RLM_CHILD_STATUS: complete' "
    "before your role work AND your handoff are done. If the native state does not arrive after several continuations, "
    "proceed with your role work using the question alone, then do your handoff protocol. "
)

def parse_final_answer(text):
    if not text:
        return None
    m = re.search(r"FINAL_ANSWER:\s*([^\n]+)", text)
    if m:
        return normalize_answer(extract_gsm8k_answer(m.group(1)) or m.group(1))
    return normalize_answer(extract_gsm8k_answer(text))

HANDOFF_RE = re.compile(r"(ADMITTED|delivered|delivery_completed|capture_armed|fallback|FAILED|failed|busy)", re.I)

def parse_handoff_status(text):
    if not text:
        return "unknown"
    t = text.lower()
    if "admitted" in t: return "admitted"
    if "fallback" in t: return "fallback"
    if "failed" in t: return "failed"
    if "busy" in t: return "busy"
    if "delivered" in t: return "delivered"
    return "unknown"

# ---------------- role prompts (repo latent-variant wording + forced handoff protocol) ----------------
def planner_prompt(question, crit_name):
    return f"""You are the Planner Agent in a sequential multi-agent system. A math question follows.

QUESTION:
{question}

Your job: design a clear, step-by-step plan for how to solve the question. Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

Then you MUST hand your plan to the next agent ({crit_name}) using the native latent channel, following this protocol EXACTLY:
1. Call occ3_handoff with receiver_role='sibling' and receiver_name='{crit_name}'. It will return 'capture_armed' - nothing sent yet.
2. In your NEXT response, output ONLY your complete plan as concise assistant text (this is the state that gets captured), then call occ3_handoff once more with receiver_role='sibling' and receiver_name='{crit_name}'. No other tool calls in that response.
3. After the tool returns, end with exactly one line: ROLE_DONE: planner"""

def critic_prompt(question, ref_name):
    return f"""You are the Critic Agent in a sequential multi-agent system. A math question follows.

QUESTION:
{question}

You will receive latent information (native state) from the Planner Agent - it will arrive as an attached native state in a later turn. " + WAIT_RULE + " When it arrives:
1. Review the plan and question.
2. Output: (1) the original plan contents, (2) constructive feedback to improve the plan. Keep it concise.

Then hand your review to the next agent ({ref_name}) via the native latent channel, following this protocol EXACTLY:
1. Call occ3_handoff with receiver_role='sibling' and receiver_name='{ref_name}'. Expect 'capture_armed'.
2. In your NEXT response, output ONLY your concise review as assistant text, then call occ3_handoff once more with receiver_role='sibling', receiver_name='{ref_name}'. No other tool calls in that response.
3. After the tool returns, end with exactly one line: ROLE_DONE: critic"""

def refiner_prompt(question, jud_name):
    return f"""You are the Refiner Agent in a sequential multi-agent system. A math question follows.

QUESTION:
{question}

You will receive latent information (native state) from the Critic Agent in a later turn. " + WAIT_RULE + " When it arrives:
1. Write a refined, improved step-by-step plan for solving the question, concise and correct. Do not produce the final answer.

Then hand your refined plan to the final solver ({jud_name}) via the native latent channel, following this protocol EXACTLY:
1. Call occ3_handoff with receiver_role='sibling' and receiver_name='{jud_name}'. Expect 'capture_armed'.
2. In your NEXT response, output ONLY your refined plan as concise assistant text, then call occ3_handoff once more with receiver_role='sibling', receiver_name='{jud_name}'. No other tool calls in that response.
3. After the tool returns, end with exactly one line: ROLE_DONE: refiner"""

def judger_prompt(question):
    return f"""You are the final solver (Judger) in a sequential multi-agent system. A math question follows.

QUESTION:
{question}

You will receive latent information (native state) from the Refiner Agent in a later turn - it may contain a plan. WAIT for it. It might contain irrelevant contents; ignore anything not helpful.

When it arrives (or after waiting), you must reason step-by-step to solve the question and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Then end your final message with exactly one line:
FINAL_ANSWER: <your final answer>"""
