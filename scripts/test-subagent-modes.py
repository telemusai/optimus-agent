#!/usr/bin/env python3
"""Real daemon/workers and execution runtimes; only provider replies are scripted."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MODES = {
    "ipython": ("ipython", "Python is the orchestration language"),
    "node": ("node", "JavaScript is the orchestration language"),
    "clang": ("clang", "C++ is the orchestration language"),
    "direct": ("bash", "Direct tools"),
}


def tool(name, arguments):
    return {"role": "assistant", "tool_calls": [{"index": 0, "id": uuid.uuid4().hex,
            "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def execution(mode, marker):
    name = MODES[mode][0]
    if mode == "direct":
        return tool(name, {"command": "printf " + marker})
    code = {
        "ipython": f"print('{marker}')",
        "node": f"console.log('{marker}')",
        "clang": f'#include <cstdio>\nint probe_{uuid.uuid4().hex} = (std::puts("{marker}"), 0);',
    }[mode]
    return tool(name, {"code": code})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("python", type=Path, help="Prepared Optimus kernel Python")
    parser.add_argument("--clang", required=True, type=Path, help="LLVM clang-repl")
    args = parser.parse_args()
    assert Path("/proc").is_dir(), "The process-isolation probe requires Linux /proc"
    repo = Path(__file__).resolve().parent.parent
    root = Path(tempfile.mkdtemp(prefix="optimus-mode-probe-"))
    for name in ("home", "agent", "sessions", "workspace", "runtime", "registry", "artifacts", "memory", "harness", "cache"):
        (root / name).mkdir()
    profile = root / "agent"
    (root / "workspace/AGENTS.md").write_text("PROJECT_MODE_PROBE: Preserve the assigned execution language.\n")
    requests, errors, frames, report = [], [], [], {"root": str(root), "passed": False, "modes": {}}
    print("Evidence:", root, flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                assert self.path == "/v1/chat/completions"
                length = int(self.headers["Content-Length"])
                assert length < 4_000_000
                body = json.loads(self.rfile.read(length))
                requests.append(body)
                assert body["model"] == "fixture-model" and body["stream"]
                users = "\n".join(json.dumps(m.get("content")) for m in body["messages"] if m["role"] == "user")
                child = "CHILD_MODE_" in users and "PARENT_MODE_" not in users
                mode = next(m for m in MODES if f"{'CHILD' if child else 'PARENT'}_MODE_{m}" in users)
                results = [m for m in body["messages"] if m["role"] == "tool"]
                outputs = "\n".join(json.dumps(m.get("content")) for m in results)
                if child:
                    prompt = "\n".join(str(m.get("content")) for m in body["messages"] if m["role"] == "system")
                    assert MODES[mode][1] in prompt, (mode, "wrong child language")
                    assert "PROJECT_MODE_PROBE" in prompt, "child lost project instructions"
                    assert body["tools"][0]["function"]["name"] == MODES[mode][0]
                    for other, (_, language) in MODES.items():
                        if other != mode and other != "direct":
                            assert language not in prompt, (mode, "stale language", other)
                    marker = f"CHILD_{'RESUME' if 'CHILD_RESUME_' in users else 'EXEC'}_OK_{mode}"
                    delta = execution(mode, marker) if marker not in outputs else {
                        "role": "assistant", "content": f"{marker}\nRLM_CHILD_STATUS: complete"}
                elif not results:
                    task = f"CHILD_MODE_{mode}: execute the language probe and finish"
                    if mode == "ipython":
                        delta = tool("ipython", {"code": f"h = await rlm({task!r}, name='mode-child-{mode}')\nprint('PARENT_SPAWNED', h.session_dir)"})
                    else:
                        delta = tool("subagent", {"action": "spawn", "name": f"mode-child-{mode}", "prompt": task})
                elif len(results) == 1:
                    if mode == "ipython":
                        delta = tool("ipython", {"code": "import json\nr = await rlm.collect(timeout_ms=30000)\nprint('PARENT_COLLECTED', json.dumps([{'status':c.status,'answer':c.answer_preview,'dir':str(c.session_dir)} for c in r]))"})
                    else:
                        delta = tool("subagent", {"action": "collect", "timeout_ms": 30000})
                else:
                    delta = {"role": "assistant", "content": "PARENT_DONE"}
                def chunk(value, finish=None):
                    return json.dumps({"id": "mode-fixture", "object": "chat.completion.chunk", "created": 1,
                        "model": "fixture-model", "choices": [{"index": 0, "delta": value, "finish_reason": finish}]})
                output = (f"data: {chunk(delta)}\n\ndata: {chunk({}, 'tool_calls' if 'tool_calls' in delta else 'stop')}\n\ndata: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)
            except Exception as error:
                errors.append(repr(error))
                self.send_error(500, repr(error))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (profile / "settings.json").write_text(json.dumps({"defaultProvider": "local-mode-fixture",
        "defaultModel": "fixture-model", "defaultThinkingLevel": "off", "retry": {"enabled": False},
        "telemetryEnabled": False, "compaction": {"enabled": False}, "autoRefine": {"enabled": False}}))
    (profile / "models.json").write_text(json.dumps({"providers": {"local-mode-fixture": {
        "baseUrl": f"http://127.0.0.1:{server.server_port}/v1", "apiKey": "synthetic-mode-key",
        "api": "openai-completions", "models": [{"id": "fixture-model", "name": "Mode fixture",
        "reasoning": False, "input": ["text"], "contextWindow": 65536, "maxTokens": 2048,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}))
    env = {"PATH": os.environ["PATH"], "HOME": str(root / "home"), "USERPROFILE": str(root / "home"),
        "TMPDIR": str(root / "runtime"), "XDG_CONFIG_HOME": str(root / "home/config"),
        "XDG_CACHE_HOME": str(root / "cache"), "PRIME_AGENT_CODING_AGENT_DIR": str(profile),
        "PRIME_AGENT_SESSION_DIR": str(root / "sessions"), "PRIME_AGENT_SESSION_ARTIFACTS_DIR": str(root / "artifacts"),
        "PRIME_AGENT_MEMORY_DIR": str(root / "memory"), "PRIME_AGENT_HARNESS_DIR": str(root / "harness"),
        "PRIME_AGENT_INTERNAL_DAEMON_SUPERVISOR_REGISTRY_DIR": str(root / "registry"),
        "PRIME_AGENT_KERNEL_PYTHON": str(args.python.absolute()), "OPTIMUS_CLANG_REPL": str(args.clang.absolute()),
        "PI_PACKAGE_DIR": str(repo / "resources/agent"), "PI_CODING_AGENT_MODULE_DIR": str(repo / "resources/agent"),
        "PI_OFFLINE": "1", "OPTIMUS_CLAUDE_CODE_AUTH": "0", "OPTIMUS_KIRO_CLI_AUTH": "0",
        "NO_PROXY": "127.0.0.1,localhost", "TERM": "dumb", "PYTHONDONTWRITEBYTECODE": "1"}
    processes, conn, reader, protocol = [], None, None, None
    sock = root / "runtime/daemon.sock"

    def owned():
        result = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                if f"PRIME_AGENT_CODING_AGENT_DIR={profile}".encode() in (path / "environ").read_bytes().split(b"\0"):
                    result.append(int(path.name))
            except OSError:
                pass
        return result

    def stop():
        nonlocal conn, reader
        if reader:
            reader.close()
        if conn:
            conn.close()
        reader, conn = None, None
        for pid in owned():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        deadline = time.monotonic() + 5
        while owned() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not owned(), "isolated workers survived cleanup"
        sock.unlink(missing_ok=True)

    def start(label):
        nonlocal conn, reader, protocol
        with (root / f"daemon-{label}.log").open("w") as log:
            process = subprocess.Popen([str(args.binary.resolve()), "--mode", "daemon", "--offline",
                "--daemon-socket", str(sock)], cwd=root / "workspace", env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        processes.append(process)
        deadline = time.monotonic() + 30
        while not sock.exists():
            assert process.poll() is None and time.monotonic() < deadline, "daemon failed to start"
            time.sleep(0.05)
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(60)
        conn.connect(str(sock))
        reader = conn.makefile("rb")
        hello = json.loads(reader.readline())
        protocol = hello["protocol"]

    def request(command):
        ident = uuid.uuid4().hex
        conn.sendall((json.dumps({"type": "command", "id": ident, "protocol": protocol,
                                 "command": dict(command, id=ident)}) + "\n").encode())
        while True:
            line = reader.readline()
            assert line, "daemon disconnected"
            frame = json.loads(line)
            frames.append(frame)
            if frame.get("type") == "response" and frame.get("id") == ident:
                assert frame.get("success"), frame.get("error", frame)
                return frame.get("data", {})

    config = {"cwd": str(root / "workspace"), "agentDir": str(profile), "sessionDir": str(root / "sessions"),
        "provider": "local-mode-fixture", "model": "fixture-model", "thinking": "off",
        "noSkills": True, "noExtensions": True, "noPromptTemplates": True, "telemetryDisabled": True}
    def create(path=None):
        data = request({"type": "create", "config": config, **({"sessionPath": str(path)} if path else {})})
        active = data.get("activeSessionId", data.get("id"))
        assert active, data
        request({"type": "attach", "activeSessionId": active})
        return active

    def messages(active):
        data = request({"type": "get_messages", "activeSessionId": active})
        return data["messages"] if isinstance(data, dict) else data

    try:
        start("initial")
        for mode in MODES:
            active = create()
            request({"type": "prompt_and_wait", "activeSessionId": active, "message": f"/mode {mode}"})
            state = request({"type": "get_connection_state", "activeSessionId": active})
            expected = state["activeToolNames"]
            request({"type": "prompt_and_wait", "activeSessionId": active, "message": f"PARENT_MODE_{mode}: spawn a child and collect its result"})
            history = messages(active)
            results = [m for m in history if m.get("role") == "toolResult"]
            assert len(results) == 2 and all(not m.get("isError") for m in results), results
            text = "\n".join(b.get("text", "") for b in results[-1]["content"])
            collected = json.loads(text.split("PARENT_COLLECTED ", 1)[1]) if mode == "ipython" else json.loads(text)["results"]
            assert len(collected) == 1 and collected[0]["status"] == "done", collected
            assert f"CHILD_EXEC_OK_{mode}" in json.dumps(collected[0]), collected
            candidates = []
            for path in root.rglob("*.jsonl"):
                text = path.read_text()
                if f"CHILD_MODE_{mode}" in text and f"PARENT_MODE_{mode}" not in text and '"type":"session"' in text:
                    candidates.append(path)
            assert len(candidates) == 1, candidates
            path = candidates[0]
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            assert any(e.get("customType") == "execution_mode" and e.get("data", {}).get("mode") == mode for e in entries), path
            child_requests = [r for r in requests if f"CHILD_MODE_{mode}" in json.dumps(r["messages"]) and f"PARENT_MODE_{mode}" not in json.dumps(r["messages"])]
            assert child_requests and all([t["function"]["name"] for t in r["tools"]] == expected for r in child_requests)
            request({"type": "prompt_and_wait", "activeSessionId": active, "message": "/mode cycle"})
            report["modes"][mode] = {"tools": expected, "child_file": str(path), "spawn_collect": True}
            request({"type": "detach", "activeSessionId": active})
            print("PASS spawn, execution, collect:", mode, flush=True)
        stop()
        start("restarted")
        for mode, data in report["modes"].items():
            active = create(Path(data["child_file"]))
            state = request({"type": "get_connection_state", "activeSessionId": active})
            assert state["activeToolNames"] == data["tools"], (mode, state["activeToolNames"])
            request({"type": "prompt_and_wait", "activeSessionId": active, "message": f"CHILD_RESUME_{mode}: execute a probe in the restored language"})
            history = messages(active)
            results = [m for m in history if m.get("role") == "toolResult"]
            assert results and not results[-1].get("isError") and f"CHILD_RESUME_OK_{mode}" in json.dumps(results[-1]), results
            data["cold_resume"] = True
            request({"type": "detach", "activeSessionId": active})
            print("PASS cold resume:", mode, flush=True)
        assert not errors, errors
        report["passed"] = True
    finally:
        stop()
        server.shutdown()
        server.server_close()
        report["provider_errors"] = errors
        for name, value in (("report", report), ("requests", requests), ("frames", frames)):
            (root / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    print("PASS: IPython, Node, Clang and Direct child modes survive parent switches and daemon restart", flush=True)


if __name__ == "__main__":
    main()
