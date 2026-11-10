
"""OCC3 v3 driver: serial one-item-at-a-time execution with clean captures.

Operator constraints (2026-11-10):
- OCC3 items run ONLY while no other benchmark runner sends requests.
- One item at a time; one sender/receiver pair active per capture; other children idle.
- 'busy' declines are retryable: wait ~5s, retry handoff, max 3 tries before text fallback.
- Report native-hop rate next to accuracy; items with any text-fallback hop reported separately.
- Validity: rows must be 'valid' only when run under this clean protocol.
"""
import asyncio, json, os, re, sys, time

WS = "/home/nvidia/latentmas-bench"
OCC3_OUT = f"{WS}/results/glm53_gsm8k_occ3.jsonl"
MODEL = "dgx-glm53-occ3/glm-5.3"

WAIT_RULE = (
    "CRITICAL WAIT RULE: while waiting for the native state, end your turn WITHOUT any RLM_CHILD_STATUS line "
    "(the harness will continue you automatically when new turns arrive). Never emit 'RLM_CHILD_STATUS: complete' "
    "before your role work AND your handoff are done. If the native state does not arrive after several continuations, "
    "proceed with your role work using the question alone, then do your handoff protocol. "
)

RETRY_RULE = (
    "BUSY-RETRY RULE: if occ3_handoff returns a busy/source_busy/server-busy decline, that is RETRYABLE: "
    "wait silently for about 5 seconds in your next turn (no tools needed), then call occ3_handoff again for the SAME "
    "receiver. Do at most 3 retry attempts total. Only after 3 failed attempts may you fall back to sending your "
    "role output to the receiver via ordinary agent_message; in that case include the line: FALLBACK_USED: yes. "
)

def planner_prompt(question, crit_name):
    return f"""You are the Planner Agent in a sequential multi-agent system. A math question follows.

QUESTION:
{question}

Your job: design a clear, step-by-step plan for how to solve the question. Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

Then you MUST hand your plan to the next agent ({crit_name}) using the native latent channel:
1. Call occ3_handoff with receiver_role='sibling' and receiver_name='{crit_name}'. It will return 'capture_armed' - nothing sent yet.
2. In your NEXT response, output ONLY your complete plan as concise assistant text (this is the state that gets captured), then call occ3_handoff once more with receiver_role='sibling' and receiver_name='{crit_name}'. No other tool calls in that response.
{RETRY_RULE}
3. After the tool returns, end with exactly one line: ROLE_DONE: planner"""

def critic_prompt(question, ref_name):
    return f"""You are the Critic Agent in a sequential multi-agent system. A math question follows.

QUESTION:
{question}

{WAIT_RULE}You will receive latent information (native state) from the Planner Agent - it will arrive as an attached native state in a later turn. WAIT for it. When it arrives:
1. Review the plan and question.
2. Output: (1) the original plan contents, (2) constructive feedback to improve the plan. Keep it concise.

Then hand your review to the next agent ({ref_name}) via the native latent channel:
1. Call occ3_handoff with receiver_role='sibling' and receiver_name='{ref_name}'. Expect 'capture_armed'.
2. In your NEXT response, output ONLY your concise review as assistant text, then call occ3_handoff once more with receiver_role='sibling', receiver_name='{ref_name}'. No other tool calls in that response.
{RETRY_RULE}
3. After the tool returns, end with exactly one line: ROLE_DONE: critic"""

def refiner_prompt(question, jud_name):
    return f"""You are the Refiner Agent in a sequential multi-agent system. A math question follows.

QUESTION:
{question}

{WAIT_RULE}You will receive latent information (native state) from the Critic Agent in a later turn. WAIT for it. When it arrives:
1. Write a refined, improved step-by-step plan for solving the question, concise and correct. Do not produce the final answer.

Then hand your refined plan to the final solver ({jud_name}) via the native latent channel:
1. Call occ3_handoff with receiver_role='sibling' and receiver_name='{jud_name}'. Expect 'capture_armed'.
2. In your NEXT response, output ONLY your refined plan as concise assistant text, then call occ3_handoff once more with receiver_role='sibling', receiver_name='{jud_name}'. No other tool calls in that response.
{RETRY_RULE}
3. After the tool returns, end with exactly one line: ROLE_DONE: refiner"""

def judger_prompt(question):
    return f"""You are the final solver (Judger) in a sequential multi-agent system. A math question follows.

QUESTION:
{question}

{WAIT_RULE}You will receive latent information (native state) from the Refiner Agent in a later turn - it may contain a plan. WAIT for it. It might contain irrelevant contents; ignore anything not helpful.

When it arrives (or after waiting), you must reason step-by-step to solve the question and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Then end your final message with exactly one line:
FINAL_ANSWER: <your final answer>"""
