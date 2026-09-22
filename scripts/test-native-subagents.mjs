#!/usr/bin/env node
// Real supervisor, Rust worker and Python REPL; only model responses are scripted.
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const [binaryArg, pythonArg] = process.argv.slice(2);
assert(binaryArg && pythonArg, "Usage: node scripts/test-native-subagents.mjs <optimus-rust> <private-python>");
assert.equal(process.platform, "linux", "This process-isolation probe currently requires Linux /proc.");
const binary = fs.realpathSync(binaryArg);
const python = path.resolve(pythonArg);
const repo = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const scratch = path.join(repo, ".port-env/tmp");
fs.mkdirSync(scratch, { recursive: true });
const root = fs.mkdtempSync(path.join(scratch, "native-subagents-"));
for (const dir of [
	"home",
	"tmp",
	"agent",
	"sessions",
	"workspace",
	"registry",
	"artifacts",
	"memory",
	"harness",
	"cache",
]) {
	fs.mkdirSync(path.join(root, dir));
}
// Short pathname for AF_UNIX sockets; all bytes still live under private scratch.
const tmpFd = fs.openSync(path.join(root, "tmp"), "r");
const tmp = `/proc/${process.pid}/fd/${tmpFd}`;
const socketPath = `${tmp}/daemon.sock`;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const requests = [];
const frames = [];
const captures = [];
const commands = [];
const errors = [];
const sockets = new Set();
const spawnCode = `import asyncio, json
print('PYTHON_READY', flush=True)
handles = await asyncio.gather(*[rlm('NATIVE_CHILD_PROBE: send the parent a reply', name='native-probe-'+str(i)) for i in range(2)])
print('NATIVE_HANDLES', json.dumps([h.rlm_child_id for h in handles]), flush=True)
for attempt in range(200):
    children = await rlm.list_subagents()
    if len(children) == 2 and all(c.status == 'completed' for c in children):
        break
    await asyncio.sleep(0.1)
else:
    raise RuntimeError('Native children did not complete')
print('NATIVE_CHILDREN_COMPLETED', flush=True)`;
const replyCode = `import rlm as runtime
receipt = await runtime.host_request('agent_message.send', {'message':'NATIVE_CHILD_REPLY', 'receiver_role':'parent'})
print('PARENT_REPLY_SENT', receipt)`;
const resumeCode = `children = await rlm.list_subagents()
assert len(children) == 2, children
assert all(c.status == 'completed' for c in children), children
print('RESUME_NATIVE_CHILDREN_OK')`;
function tool(code, id) {
	return {
		role: "assistant",
		tool_calls: [
			{ index: 0, id, type: "function", function: { name: "ipython", arguments: JSON.stringify({ code }) } },
		],
	};
}
const server = http.createServer(async (req, res) => {
	try {
		assert.equal(req.url, "/v1/chat/completions");
		let text = "";
		for await (const chunk of req) {
			text += chunk;
			assert(text.length < 2_000_000);
		}
		const body = JSON.parse(text);
		requests.push(body);
		assert.equal(body.model, "fixture-model");
		assert.equal(body.stream, true);
		const users = body.messages.filter((m) => m.role === "user").map((m) => JSON.stringify(m.content));
		const child = users.some((m) => m.includes("NATIVE_CHILD_PROBE"));
		const resuming = users.some((m) => m.includes("RESUME_PROBE"));
		const toolResults = body.messages.filter((m) => m.role === "tool");
		let delta;
		if (child && toolResults.length === 0) delta = tool(replyCode, "native-reply");
		else if (child) {
			await sleep(500);
			delta = { role: "assistant", content: "NATIVE_CHILD_OK\nRLM_CHILD_STATUS: complete" };
		} else if (resuming && !toolResults.some((m) => JSON.stringify(m).includes("RESUME_NATIVE_CHILDREN_OK")))
			delta = tool(resumeCode, "resume-probe");
		else if (toolResults.length === 0) delta = tool(spawnCode, "native-spawn");
		else delta = { role: "assistant", content: "PARENT_PROBE_OK" };
		const chunk = (value, finish) => ({
			id: "native-fixture",
			object: "chat.completion.chunk",
			created: 1,
			model: "fixture-model",
			choices: [{ index: 0, delta: value, finish_reason: finish }],
		});
		const output = `data: ${JSON.stringify(chunk(delta, null))}\n\ndata: ${JSON.stringify(chunk({}, delta.tool_calls ? "tool_calls" : "stop"))}\n\ndata: [DONE]\n\n`;
		res.writeHead(200, { "Content-Type": "text/event-stream", "Content-Length": Buffer.byteLength(output) });
		res.end(output);
	} catch (error) {
		errors.push(String(error));
		res.writeHead(500);
		res.end(String(error));
	}
});
server.on("connection", (socket) => {
	sockets.add(socket);
	socket.on("close", () => sockets.delete(socket));
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const env = {
	PATH: process.env.PATH,
	HOME: `${root}/home`,
	USERPROFILE: `${root}/home`,
	TMPDIR: tmp,
	TEMP: `${root}/tmp`,
	TMP: `${root}/tmp`,
	XDG_CONFIG_HOME: `${root}/home/config`,
	XDG_CACHE_HOME: `${root}/cache`,
	PRIME_AGENT_CODING_AGENT_DIR: `${root}/agent`,
	PRIME_AGENT_SESSION_DIR: `${root}/sessions`,
	PRIME_AGENT_SESSION_ARTIFACTS_DIR: `${root}/artifacts`,
	PRIME_AGENT_MEMORY_DIR: `${root}/memory`,
	PRIME_AGENT_HARNESS_DIR: `${root}/harness`,
	PRIME_AGENT_INTERNAL_DAEMON_SUPERVISOR_REGISTRY_DIR: `${root}/registry`,
	PRIME_AGENT_KERNEL_PYTHON: python,
	PI_PACKAGE_DIR: `${repo}/resources/agent`,
	PI_CODING_AGENT_MODULE_DIR: `${repo}/resources/agent`,
	PI_OFFLINE: "1",
	NO_PROXY: "127.0.0.1,localhost",
	TERM: "dumb",
	PYTHONDONTWRITEBYTECODE: "1",
};
fs.writeFileSync(
	`${root}/agent/settings.json`,
	JSON.stringify({
		defaultProvider: "local-native-fixture",
		defaultModel: "fixture-model",
		defaultThinkingLevel: "off",
		telemetry: { enabled: false },
		autoCompaction: { enabled: false },
		quietStartup: true,
		onboardingShown: true,
	}),
);
fs.writeFileSync(
	`${root}/agent/models.json`,
	JSON.stringify({
		providers: {
			"local-native-fixture": {
				baseUrl: `http://127.0.0.1:${server.address().port}/v1`,
				apiKey: "local-fixture-not-a-secret",
				api: "openai-completions",
				models: [
					{
						id: "fixture-model",
						name: "Local fixture",
						reasoning: false,
						input: ["text"],
						contextWindow: 65536,
						maxTokens: 2048,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					},
				],
			},
		},
	}),
);
function launch(label, args) {
	const stdout = fs.openSync(`${root}/${label}.stdout`, "w");
	const stderr = fs.openSync(`${root}/${label}.stderr`, "w");
	const child = spawn(binary, [...args, "--daemon-socket", socketPath], {
		env,
		cwd: `${root}/workspace`,
		detached: true,
		stdio: ["ignore", stdout, stderr],
	});
	fs.closeSync(stdout);
	fs.closeSync(stderr);
	commands.push(child);
	child.on("error", (error) => errors.push(String(error)));
	return child;
}
async function waitFor(check, description, timeout = 30_000) {
	const deadline = Date.now() + timeout;
	while (!check()) {
		assert(Date.now() < deadline, `Timed out: ${description}. Evidence: ${root}`);
		await sleep(50);
	}
}
const tmuxArgs = ["-S", `${tmp}/tmux.sock`];
const tuiEnv = { ...env, TERM: "xterm-256color", SHELL: "/bin/bash" };
function tmux(args) {
	return execFileSync("tmux", [...tmuxArgs, ...args], {
		env: tuiEnv,
		cwd: `${root}/workspace`,
		encoding: "utf8",
		timeout: 5000,
	});
}
let tuiStarted = false;
let control;
try {
	const daemon = launch("daemon", ["--mode", "daemon", "--offline"]);
	await waitFor(() => fs.existsSync(socketPath) || daemon.exitCode !== null, "daemon socket");
	assert.equal(daemon.exitCode, null);
	control = net.createConnection(socketPath);
	let buffered = "";
	control.on("data", (data) => {
		buffered += data;
		let newline;
		while ((newline = buffered.indexOf("\n")) !== -1) {
			const line = buffered.slice(0, newline);
			buffered = buffered.slice(newline + 1);
			try {
				frames.push(JSON.parse(line));
			} catch (error) {
				errors.push(String(error));
			}
		}
	});
	control.on("error", (error) => errors.push(String(error)));
	await waitFor(() => frames.some((f) => f.type === "daemon_hello"), "daemon handshake");
	control.write(`${JSON.stringify({ type: "roster_subscribe", id: "roster" })}\n`);
	await waitFor(() => frames.some((f) => f.type === "response" && f.id === "roster"), "roster subscription");
	const start = Date.now();
	const cli = launch("spawn", [
		"--print",
		"--mode",
		"json",
		"--offline",
		"--provider",
		"local-native-fixture",
		"--model",
		"fixture-model",
		"Launch two native agents and wait for their replies",
	]);
	await waitFor(() => cli.exitCode !== null, "native children and parent completion", 45_000);
	assert.equal(cli.exitCode, 0, fs.readFileSync(`${root}/spawn.stderr`, "utf8"));
	const output = fs.readFileSync(`${root}/spawn.stdout`, "utf8");
	const events = output
		.trim()
		.split("\n")
		.map((line) => JSON.parse(line));
	const result = events.find((e) => e.type === "tool_execution_end" && e.toolCallId === "native-spawn");
	assert(result, "Missing parent tool result");
	assert.equal(result.isError, false, JSON.stringify(result));
	assert(JSON.stringify(result).includes("NATIVE_CHILDREN_COMPLETED"), JSON.stringify(result));
	assert(output.includes("NATIVE_CHILD_REPLY"), "Parent never received a child reply");
	assert(output.includes("PARENT_PROBE_OK"), "Parent did not finish");
	const sessionFiles = fs.readdirSync(`${root}/sessions`).filter((name) => name.endsWith(".jsonl"));
	assert.equal(sessionFiles.length, 1);
	const resumed = launch("resume", [
		"--print",
		"--mode",
		"json",
		"--offline",
		"--resume",
		`${root}/sessions/${sessionFiles[0]}`,
		"RESUME_PROBE: verify the existing two native children",
	]);
	await waitFor(() => resumed.exitCode !== null, "resumed native child registry", 30_000);
	assert.equal(resumed.exitCode, 0);
	const resumeOutput = fs.readFileSync(`${root}/resume.stdout`, "utf8");
	const resumeEvents = resumeOutput
		.trim()
		.split("\n")
		.map((line) => JSON.parse(line));
	const resumeResult = resumeEvents.find((e) => e.type === "tool_execution_end" && e.toolCallId === "resume-probe");
	assert(resumeResult && resumeResult.isError === false, JSON.stringify(resumeResult));
	assert(JSON.stringify(resumeResult).includes("RESUME_NATIVE_CHILDREN_OK"));
	tmux([
		"new-session",
		"-d",
		"-s",
		"probe",
		"-x",
		"120",
		"-y",
		"40",
		binary,
		"--offline",
		"--daemon-socket",
		socketPath,
		"--provider",
		"local-native-fixture",
		"--model",
		"fixture-model",
	]);
	tuiStarted = true;
	let screen = "";
	await waitFor(() => {
		screen = tmux(["capture-pane", "-p", "-t", "probe"]);
		return screen.includes("0 agents");
	}, "always visible idle footer");
	fs.writeFileSync(`${root}/tui-empty.txt`, screen);
	tmux(["set-buffer", "-b", "probe-prompt", "Launch two native agents and wait for their replies"]);
	tmux(["paste-buffer", "-p", "-b", "probe-prompt", "-t", "probe"]);
	await waitFor(
		() => tmux(["capture-pane", "-p", "-t", "probe"]).includes("Launch two native agents"),
		"pasted prompt",
	);
	tmux(["send-keys", "-t", "probe", "Enter"]);
	let settledSince;
	await waitFor(
		() => {
			screen = tmux(["capture-pane", "-p", "-t", "probe"]);
			if (captures.at(-1) !== screen) captures.push(screen);
			const latest = new Map();
			for (const frame of frames) {
				if (frame.type !== "roster_update") continue;
				for (const row of frame.changed ?? []) latest.set(row.agentId, row);
				for (const id of frame.removed ?? []) latest.delete(id);
			}
			// The top-level activity label also waits for a summarizer verdict in TypeScript.
			// Assert actual run quiescence independently of that semantic classification.
			const parentQuiescent = [...latest.values()].some(
				(row) =>
					row.summary.runtimeKind === "top-level" &&
					row.summary.isSessionActive === false &&
					row.summary.isStreaming === false &&
					row.summary.hasRunningRlmChildren === false,
			);
			const settled =
				parentQuiescent &&
				screen.includes("PARENT_PROBE_OK") &&
				/2 agents[\s\S]*2 idle/.test(screen) &&
				screen.includes("Agent message received") &&
				screen.includes("native-probe-0") &&
				screen.includes("native-probe-1") &&
				!/Waiting ·|Waiting for [0-9]+ subagents?/.test(screen);
			settledSince = settled ? (settledSince ?? Date.now()) : undefined;
			return settledSince !== undefined && Date.now() - settledSince >= 500;
		},
		"TUI completion and two idle native agents",
		45_000,
	);
	fs.writeFileSync(`${root}/tui-final.txt`, screen);
	fs.writeFileSync(`${root}/tui-captures.json`, JSON.stringify(captures, null, 2));
	assert(
		captures.some((s) => /[12] agents?[\s\S]*[12] running/.test(s)),
		"No running native agent count was rendered",
	);
	assert(!/Waiting ·|Waiting for [0-9]+ subagents?/.test(screen), "TUI remained Waiting after completion");
	assert(
		frames.some((f) => f.type === "roster_update"),
		"No public roster update was delivered",
	);
	assert(
		screen.includes("Agent message received") &&
			screen.includes("native-probe-0") &&
			screen.includes("native-probe-1"),
		"Child reply notices were not rendered",
	);
	assert.equal(errors.length, 0, errors.join("\n"));
	console.log(
		`PASS: native Python admission, two concurrent native children, explicit parent replies, completion, resume and live TUI counts (${Date.now() - start}ms)`,
	);
	console.log(`Evidence: ${root}`);
} finally {
	fs.writeFileSync(`${root}/tui-captures.json`, JSON.stringify(captures, null, 2));
	fs.writeFileSync(`${root}/requests.json`, JSON.stringify(requests, null, 2));
	fs.writeFileSync(`${root}/roster.json`, JSON.stringify(frames, null, 2));
	control?.destroy();
	if (tuiStarted) {
		try {
			tmux(["kill-server"]);
		} catch {
			/* Already exited. */
		}
	}
	// Workers have their own process groups. Match our private profile, never a PID alone.
	for (const entry of fs.readdirSync("/proc")) {
		if (!/^\d+$/.test(entry)) continue;
		try {
			const environment = fs.readFileSync(`/proc/${entry}/environ`).toString().split("\0");
			if (environment.includes(`PRIME_AGENT_CODING_AGENT_DIR=${root}/agent`)) process.kill(Number(entry), "SIGKILL");
		} catch {
			/* Process already exited or belongs to another user. */
		}
	}
	for (const child of commands) {
		try {
			child.kill("SIGKILL");
		} catch {
			/* Already exited. */
		}
	}
	for (const socket of sockets) socket.destroy();
	await new Promise((resolve) => server.close(resolve));
	fs.closeSync(tmpFd);
}
