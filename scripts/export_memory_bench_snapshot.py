#!/usr/bin/env python3
"""Export explicitly selected, stopped memory_bench snapshots without a runtime.

Snapshot layout: RUN/envs/ENV/{agent,workspace,session-artifacts}. Only the
selected project's native document, host-id, memory settings, and opted-in
harness/cache files are read. No registry discovery, profile defaults, model or
credential loading, env.jsonl/store_snapshot fallback, network, or LLM calls.

Examples (all paths are explicit; output directories must not already exist):
  python scripts/export_memory_bench_snapshot.py snapshot --run-dir frozen/run \
    --run-id r --env-id e --project-id project_example --code-revision REV \
    --capture-stage final_state --frozen --persisted-settings-for-current-replay \
    --allow-source-root frozen/run/envs/e/workspace --output-dir exports/cut
  python scripts/export_memory_bench_snapshot.py queries --input questions.jsonl \
    --output-dir exports/questions

The snapshot bundle contains the exact memory_recall_probe fixture and a
separate provenance.json. Persisted settings are REQUESTED settings, not proof
of historical effective settings (including the old apply_settings bug).
Alternatively, --effective-memory-settings requires a full native settings
object and --settings-provenance; it is a caller-supplied override, not runtime
verification. The probe always resolves settings using its CURRENT native code
and only replays lexical retrieval, never historical LLM feature activation.

--frozen attests that the caller stopped writes. Reads detect ordinary file
changes, but do not lock a runtime or establish a transactional capture. Do not
run against a concurrently modified or adversarially changing directory tree.
Source files must be local regular UTF-8 files under --allow-source-root. All
symlinks/reparse points, hard-linked sources, unsafe paths, unreadable/non-UTF-8/oversized files, and
unsupported file URLs fail closed; only actual absence becomes content:null.
projectPath is remapped through the frozen workspace, not the original root
label. --source-project-root overrides that mapping explicitly. A copied run
can preserve its original identity with --project-root and --project-alias.

The independent queries command copies only qid/variant/query/options from
JSONL (question is accepted instead of query). Answers, gold, evidence, and all
other top-level fields are excluded. Benchmark rows with run_id/env_id require
explicit selectors. No query/answer artifact is read by the snapshot command.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import unquote, urlsplit

FIXTURE_SCHEMA = "optimus-memory-recall-fixture/v1"
EXPORT_SCHEMA = "optimus-memory-recall-export/v1"
KINDS = ("memory", "prompt", "skill", "subagent")
EXTRA_SCOPES = ("global", "session", "shared")
BOOL_SETTINGS = ("recall", "learning", "recallQueryDistillation", "recallRerank")
LIMITS = {"maxRecallChars": (0, 32000), "maxRecallEntries": (0, 50),
          "maxExtractionTokens": (256, 32000), "maxImportBytes": (1024, 128 * 1024 * 1024),
          "maxImportChunkChars": (1000, 80000), "maxImportChunksPerRun": (1, 64)}
SETTING_KEYS = set(BOOL_SETTINGS) | set(LIMITS) | {"importInstructions", "shared"}
SOURCE_LIMIT = 32 * 1024 * 1024


class ExportError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ExportError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def nonempty(value, label):
    require(isinstance(value, str) and value.strip(), f"{label} must be a nonempty string")
    require("\0" not in value, f"{label} contains a NUL")
    return value


def object_value(value, label):
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def reject_constant(_value):
    raise ExportError("non-finite JSON number")


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), "non-finite JSON number")
    return result


def parse_json(text, label):
    try:
        return json.loads(text, object_pairs_hook=no_duplicate_keys,
                          parse_constant=reject_constant, parse_float=finite_float)
    except (ValueError, RecursionError) as error:
        # Never echo input text: a rejected query line can contain gold answers.
        raise ExportError(f"{label}: invalid JSON ({type(error).__name__})") from error


def json_bytes(value):
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                           allow_nan=False) + "\n").encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ExportError("output is not finite, valid UTF-8 JSON") from error


def safe_component(value, label):
    nonempty(value, label)
    require(re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value) is not None,
            f"unsafe {label}")
    require(value not in (".", "..") and not value.endswith((".", " ")),
            f"unsafe {label}")
    return value


def absolute_path(value):
    path = Path(value)
    require(".." not in path.parts, "path traversal is not allowed")
    # No UNC/device paths: inspecting them could itself contact a remote host.
    require(not str(path).startswith(("\\\\", "//")), "UNC/device paths are unsupported")
    if os.name == "nt":
        require(not path.drive or path.is_absolute(), "drive-relative paths are unsupported")
    path = Path(os.path.abspath(path))
    for part in path.parts[1:]:
        require("\0" not in part, "NUL in path")
        if os.name == "nt":
            require(":" not in part and not part.endswith((".", " ")), "unsafe Windows path")
            stem = part.split(".", 1)[0].upper()
            require(not re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", stem),
                    "reserved Windows path")
    return path


def checked_stat(path):
    """lstat every component; never follow symlinks or Windows reparse points."""
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate((None, *parts)):
        if part is not None:
            current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise ExportError(f"cannot inspect path: {current}") from error
        require(not stat.S_ISLNK(info.st_mode)
                and not (getattr(info, "st_file_attributes", 0)
                         & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
                f"symlink/reparse point is not allowed: {current}")
        if index < len(parts):
            require(stat.S_ISDIR(info.st_mode), f"path parent is not a directory: {current}")
    return info


def directory(value):
    path = absolute_path(value)
    info = checked_stat(path)
    require(info is not None and stat.S_ISDIR(info.st_mode), f"directory is missing: {path}")
    return path


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def fingerprint(info):
    return None if info is None else (info.st_dev, info.st_ino, info.st_size,
                                     info.st_mtime_ns, info.st_ctime_ns)


class Reader:
    def __init__(self, max_input_bytes, max_total_bytes):
        self.max_input_bytes = max_input_bytes
        self.max_total_bytes = max_total_bytes
        self.total_bytes = 0
        self.inputs = []
        self.observed = {}

    def read(self, value, role, *, optional=False, roots=None, limit=None):
        path = absolute_path(value)
        if roots is not None:
            require(any(inside(path, root) for root in roots), f"input outside allowed roots: {path}")
        before = checked_stat(path)
        if before is None:
            require(optional, f"required input is missing: {path}")
            raw = None
        else:
            require(stat.S_ISREG(before.st_mode), f"input is not a regular file: {path}")
            if role.startswith("source_file:"):
                require(before.st_nlink == 1, f"hard-linked source is outside the conservative file sandbox: {path}")
            bound = self.max_input_bytes if limit is None else min(limit, self.max_input_bytes)
            require(before.st_size <= bound, f"input is oversized: {path}")
            require(self.total_bytes + before.st_size <= self.max_total_bytes,
                    "total input byte limit exceeded")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            try:
                with os.fdopen(os.open(path, flags), "rb") as stream:
                    require(fingerprint(os.fstat(stream.fileno())) == fingerprint(before),
                            f"input changed before read: {path}")
                    raw = stream.read(bound + 1)
                    require(len(raw) <= bound, f"input is oversized: {path}")
                    require(fingerprint(os.fstat(stream.fileno())) == fingerprint(before)
                            and len(raw) == before.st_size, f"input changed during read: {path}")
            except OSError as error:
                raise ExportError(f"input is unreadable: {path}") from error
            require(fingerprint(checked_stat(path)) == fingerprint(before),
                    f"input changed during read: {path}")
            self.total_bytes += len(raw)
        key = str(path)
        observed = fingerprint(before)
        require(key not in self.observed or self.observed[key] == observed,
                f"input changed between reads: {path}")
        self.observed[key] = observed
        record = {"role": role, "path": key, "status": "missing" if raw is None else "utf8",
                  "sha256": None if raw is None else sha256(raw),
                  "size_bytes": None if raw is None else len(raw)}
        self.inputs.append(record)
        if raw is None:
            return None, record
        try:
            text = raw.decode("utf-8")
        except UnicodeError as error:
            raise ExportError(f"input is not UTF-8 (not missing): {path}") from error
        return text, record

    def json(self, value, role, **kwargs):
        text, _record = self.read(value, role, **kwargs)
        if text is None:
            return None
        parsed = parse_json(text, role)
        require(parsed is not None, f"{role}: JSON null is not a missing input")
        return parsed

    def verify(self):
        for path, expected in self.observed.items():
            require(fingerprint(checked_stat(Path(path))) == expected,
                    f"input changed since capture: {path}")


def validate_settings(value, *, complete=False):
    value = object_value(value, "memory settings")
    require(not (set(value) - SETTING_KEYS), "unsupported memory settings keys")
    if complete:
        require(set(BOOL_SETTINGS) | set(LIMITS) <= set(value),
                "effective settings must be complete, including both recall feature flags")
    for key in BOOL_SETTINGS:
        if key in value:
            require(type(value[key]) is bool, f"{key} must be boolean")
    for key, (low, high) in LIMITS.items():
        if key in value:
            require(type(value[key]) is int and low <= value[key] <= high, f"invalid {key}")
    if "importInstructions" in value:
        require(value["importInstructions"] is None or isinstance(value["importInstructions"], str),
                "invalid importInstructions")
    shared = value.get("shared")
    if shared is not None:
        object_value(shared, "shared settings")
        require(set(shared) == {"url", "tokenFile"}, "shared settings require only url/tokenFile")
        nonempty(shared["tokenFile"], "tokenFile path (never read)")
        url = nonempty(shared["url"], "shared URL")
        try:
            parsed = urlsplit(url)
            require(parsed.hostname is not None and parsed.username is None
                    and parsed.password is None and not parsed.query and not parsed.fragment
                    and (parsed.scheme == "https" or (parsed.scheme == "http"
                         and parsed.hostname in ("localhost", "127.0.0.1", "::1"))),
                    "unsafe shared URL")
        except ValueError as error:
            raise ExportError("invalid shared URL") from error
    return value


def validate_harness(value, project_id=None):
    value = object_value(value, "native harness")
    require(type(value.get("schema")) in (int, float) and value["schema"] == 1,
            "unsupported native harness schema")
    entries = object_value(value.get("entries"), "native entries")
    if project_id is None:
        require(set(entries) <= set(KINDS), "unsupported native harness kind bucket")
    else:
        require(set(entries) == set(KINDS), "native entries must contain all four kind buckets")
    require(isinstance(value.get("refinements"), list), "native refinements must be an array")
    for kind, bucket in entries.items():
        for entry_id, entry in object_value(bucket, "entry bucket").items():
            object_value(entry, "entry")
            require(entry.get("id") == entry_id and entry.get("kind") == kind,
                    "native entry ID/kind mismatch")
            for key in ("id", "title", "content", "path", "created_at", "updated_at"):
                require(isinstance(entry.get(key), str), f"entry {key} must be a string")
            require(type(entry.get("version")) is int and -(2**63) <= entry["version"] < 2**63,
                    "entry version must be an i64")
            for key in ("metadata", "reference", "arguments"):
                object_value(entry.get(key, {}), f"entry {key}")
            if project_id is not None:
                require(entry.get("metadata", {}).get("projectId") == project_id,
                        "entry belongs to another project")
    return value


def validate_document(value, project_id):
    validate_harness(value, project_id)
    metadata = object_value(value.get("memory"), "full native memory metadata (not store_snapshot)")
    require(type(metadata.get("schema")) in (int, float) and metadata["schema"] == 1
            and metadata.get("projectId") == project_id, "native memory project/schema mismatch")
    require(type(metadata.get("revision")) is int and 0 <= metadata["revision"] < 2**63,
            "invalid memory revision")
    require(isinstance(metadata.get("history"), list), "memory history must be an array")
    events = object_value(metadata.get("events"), "memory events")
    require(all(isinstance(item, str) for item in events.values()), "memory events must be strings")
    return value


def project_relative(value):
    nonempty(value, "projectPath")
    require(not any(char in value for char in ("\\", ":", "\0"))
            and all(part not in ("", ".", "..") for part in value.split("/")),
            "unsafe/nonportable projectPath")
    return value


def local_file_path(uri):
    """Conservative local file-URI subset; reject, never simulate unsupported URLs."""
    try:
        parsed = urlsplit(uri)
        require(parsed.scheme == "file" and parsed.netloc in ("", "localhost")
                and not parsed.query and not parsed.fragment and parsed.path.startswith("/")
                and not any(ord(char) < 32 or ord(char) == 127 for char in uri)
                and "\\" not in uri and not re.search(r"%(?![0-9A-Fa-f]{2})", uri),
                "unsupported file URI (only absolute local file URLs are accepted)")
        decoded = unquote(parsed.path, encoding="utf-8", errors="strict")
        require("\0" not in decoded and "\\" not in decoded and not decoded.startswith("//"),
                "unsafe file URI path")
        if os.name == "nt":
            require(re.match(r"^/[A-Za-z]:/", decoded) is not None,
                    "file URI is not an absolute Windows drive path")
            decoded = decoded[1:]
        return absolute_path(decoded)
    except (UnicodeError, ValueError) as error:
        if isinstance(error, ExportError):
            raise
        raise ExportError("unsupported file URI") from error


def source_is_sensitive(path, token_paths):
    lowered = [part.lower() for part in path.parts]
    name = path.name.lower()
    return (name in {"models.json", "auth.json", "settings.json", "credentials.json", "credentials",
                     "token", "tokens", "token.json", "tokens.json", "secrets.json", ".env"}
            or name.startswith(".env.") or path.suffix.lower() in {".pem", ".key"}
            or any(part in {".ssh", ".aws", ".azure"} for part in lowered)
            or any(path == other for other in token_paths))


def export_sources(documents, project, host_id, workspace, roots, reader, limit, token_paths):
    files, evidence, locations = [], [], {}
    for scope, document in documents:
        for kind in KINDS:
            for entry_id, entry in document["entries"].get(kind, {}).items():
                metadata = entry.get("metadata", {})
                owner = metadata.get("projectId")
                host = metadata.get("hostId")
                if (isinstance(owner, str) and owner != project["id"]) or (
                        isinstance(host, str) and host and host != host_id):
                    continue
                refs = metadata.get("sources", [])
                if not isinstance(refs, list):
                    continue
                for index, source in enumerate(refs):
                    if not isinstance(source, dict) or source.get("origin") not in (
                            "user", "assistant", "tool", "derived", "file") or not all(
                            isinstance(source.get(key), str) for key in ("id", "sha256")):
                        continue  # The native memory_source_from_value parser also skips these.
                    row = {"scope": scope, "kind": kind, "entry_id": entry_id, "source_index": index,
                           "source_id": source["id"], "recorded_sha256": source["sha256"]}
                    uri = source.get("uri")
                    relative = source.get("projectPath")
                    if source["origin"] != "file" or not isinstance(uri, str) or not uri.startswith("file:"):
                        row["status"] = "not_file_backed_native_unknown"
                        evidence.append(row)
                        continue
                    uri_path = local_file_path(uri)
                    if isinstance(relative, str):
                        relative = project_relative(relative)
                        key = ("project", relative)
                        path = absolute_path(workspace / relative)
                    else:
                        relative = None
                        key = ("uri", str(uri_path))
                        path = uri_path
                    require(not source_is_sensitive(path, token_paths), f"refusing sensitive source path: {path}")
                    if key not in locations:
                        text, record = reader.read(path, f"source_file:{len(files)}", optional=True,
                                                   roots=roots, limit=limit)
                        frozen = {"uri": uri, "content": text}
                        if relative is not None:
                            frozen["projectPath"] = relative
                        locations[key] = (len(files), record)
                        files.append(frozen)
                    file_index, record = locations[key]
                    row.update({"file_index": file_index, "status": record["status"],
                                "input_sha256": record["sha256"]})
                    evidence.append(row)
    return files, evidence


def reader_for(args):
    require(0 < args.max_input_bytes <= 1024**3 and 0 < args.max_total_bytes <= 1024**3,
            "byte limits must be between 1 byte and 1 GiB")
    return Reader(args.max_input_bytes, args.max_total_bytes)


def snapshot(args):
    require(args.frozen, "--frozen is required: stop writes before capturing")
    run = directory(args.run_dir)
    env_id = safe_component(args.env_id, "environment ID")
    require(re.fullmatch(r"project_[A-Za-z0-9_-]{1,80}", args.project_id) is not None,
            "invalid project ID")
    env = directory(run / "envs" / env_id)
    reader = reader_for(args)
    project_dir = env / "agent" / "memory" / "projects" / args.project_id
    project_memory = reader.json(project_dir / "harness_state.json", "project_memory", roots=[env])
    validate_document(project_memory, args.project_id)
    host = reader.json(env / "agent" / "memory" / "host-id.json", "host_identity", roots=[env])
    host_id = object_value(host, "host identity").get("id")
    require(isinstance(host_id, str) and re.fullmatch(r"[a-f0-9-]{36}", host_id) is not None,
            "invalid native host ID")
    global_value = reader.json(env / "agent" / "settings.json", "persisted_global_settings",
                               optional=True, roots=[env])
    requested_global = {} if global_value is None else object_value(global_value, "global settings").get("memory", {})
    requested_project = reader.json(project_dir / "settings.json", "persisted_project_settings",
                                    optional=True, roots=[env])
    requested_project = {} if requested_project is None else requested_project
    validate_settings(requested_global)
    validate_settings(requested_project)
    settings_provenance = {"persisted_requested": {"global_memory": requested_global,
                                                   "project_memory": requested_project},
                           "historical_runtime_verified": False,
                           "probe_current_resolved": "not_observed_by_exporter",
                           "llm_call_activation": "not_established; probe is lexical only"}
    effective = None
    if args.effective_memory_settings:
        nonempty(args.settings_provenance, "--settings-provenance for effective override")
        effective = reader.json(args.effective_memory_settings, "caller_effective_settings")
        validate_settings(effective, complete=True)
        global_settings, project_settings = effective, {}
        settings_provenance.update({"authority": "caller_effective", "caller_provenance": args.settings_provenance,
                                    "caller_effective": effective,
                                    "fixture_mapping": "caller_effective as global; empty project overlay"})
    else:
        require(args.persisted_settings_for_current_replay, "select an explicit settings authority")
        require(args.settings_provenance is None, "--settings-provenance requires an effective override")
        global_settings, project_settings = requested_global, requested_project
        settings_provenance.update({"authority": "persisted_requested",
                                    "historical_runtime_status": "historical_runtime_unverified",
                                    "fixture_mapping": "persisted memory-only global/project settings"})
    workspace = directory(args.source_project_root or env / "workspace")
    project = {"id": args.project_id, "root": args.project_root or str(workspace),
               "aliases": list(args.project_alias)}
    nonempty(project["root"], "original project root")
    for alias in project["aliases"]:
        nonempty(alias, "project alias")
    require(len(set(project["aliases"])) == len(project["aliases"]), "duplicate project aliases")
    included = sorted(set(args.include_scope))
    optional_states = {"global_harness": None, "session_harness": None, "shared_cache": None}
    documents = [("project", project_memory)]
    scope_files = {"global": ("global_harness", env / "agent" / "harness" / "harness_state.json"),
                   "session": ("session_harness", env / "session-artifacts" / "harness" / "harness_state.json"),
                   "shared": ("shared_cache", project_dir / "shared.json")}
    for scope in included:
        field, path = scope_files[scope]
        value = reader.json(path, field, optional=True, roots=[env])
        optional_states[field] = value
        if value is not None:
            if scope == "shared":
                object_value(value, "shared cache")
                require(value.get("schema") == 1 and isinstance(value.get("url"), str)
                        and type(value.get("revision")) is int and type(value.get("connected")) is bool
                        and isinstance(value.get("pending"), list), "invalid native shared cache")
                state = validate_document(value.get("state"), args.project_id)
            else:
                state = validate_harness(value)
            documents.append((scope, state))
    roots = [directory(path) for path in args.allow_source_root]
    require(0 < args.max_source_bytes <= SOURCE_LIMIT, "source limit must be between 1 and 32 MiB")
    token_paths = []
    for settings in (requested_global, requested_project, effective or {}):
        token = (settings.get("shared") or {}).get("tokenFile")
        if token:
            # Do not resolve or open token paths, including relative tokenFile values.
            token_paths.extend([absolute_path(token), absolute_path(env / "agent" / token)])
    files, source_evidence = export_sources(documents, project, host_id, workspace, roots, reader,
                                            args.max_source_bytes, token_paths)
    lineage = {"run_id": nonempty(args.run_id, "run ID"), "env_id": env_id,
               "capture_stage": nonempty(args.capture_stage, "capture stage"),
               "code_revision": nonempty(args.code_revision, "code revision")}
    fixture = {"schema": FIXTURE_SCHEMA, "lineage": lineage, "project": project, "host_id": host_id,
               "memory": project_memory, "global_memory_settings": global_settings,
               "project_memory_settings": project_settings, **optional_states, "files": files}
    manifest = {"schema": EXPORT_SCHEMA, "kind": "snapshot", "lineage": lineage,
                "capture_authority": "caller_attested_frozen_native_files; no transactional lock",
                "historical_per_question_reconstruction": False, "settings": settings_provenance,
                "selection": {"run_dir": str(run), "environment_dir": str(env),
                              "included_corpora": ["project", *included],
                              "omitted_corpora": sorted(set(EXTRA_SCOPES) - set(included)),
                              "host_entries": "native host identity filter within selected corpora",
                              "corpus_parity": "caller-selected; omitted corpora can change native IDF",
                              "source_project_root": str(workspace),
                              "project_identity_authority": "caller-selected ID/root/aliases; no git or registry discovery",
                              "project_root_authority": ("caller_supplied" if args.project_root else
                                                         "frozen_source_root_default; historical_root_unverified"),
                              "aliases_authority": "caller_supplied_or_empty; not_discovered",
                              "allowed_source_roots": [str(path) for path in roots]},
                "inputs": reader.inputs, "sources": source_evidence}
    return {"fixture.json": json_bytes(fixture)}, manifest, reader, [run, workspace, *roots]


def sanitized_query(value, number):
    object_value(value, f"query line {number}")
    require(not ("query" in value and "question" in value), "ambiguous query/question fields")
    query = value.get("query", value.get("question"))
    require(isinstance(query, str), f"query line {number} requires a query/question string")
    result = {"qid": nonempty(value.get("qid"), "qid"),
              "variant": nonempty(value.get("variant", "base"), "variant"), "query": query}
    if "options" in value:
        options = object_value(value["options"], "query options")
        require(set(options) <= {"include_inactive", "scope", "recall", "max_recall_chars", "max_recall_entries"},
                "unsupported query options (answer/evidence/options are never copied)")
        for key in ("include_inactive", "recall"):
            if key in options:
                require(type(options[key]) is bool or (key == "recall" and options[key] is None),
                        f"invalid query {key}")
        require(options.get("scope") in (None, "project", "host", "session", "global", "shared"),
                "invalid query scope")
        for key, maximum in (("max_recall_chars", 32000), ("max_recall_entries", 50)):
            if options.get(key) is not None:
                require(type(options[key]) is int and 0 <= options[key] <= maximum,
                        f"invalid query {key}")
        result["options"] = options
    return result


def queries(args):
    reader = reader_for(args)
    text, input_record = reader.read(args.input, "explicit_query_input")
    rows, lineage, seen = [], [], set()
    input_lines = text.split("\n")
    for number, line in enumerate(input_lines, 1):
        if number < len(input_lines) and line.endswith("\r"):
            line = line[:-1]
        if not line.strip():
            continue
        value = object_value(parse_json(line, f"query line {number}"), "query row")
        selected = True
        for field in ("run_id", "env_id"):
            wanted = getattr(args, field)
            if field in value or wanted is not None:
                require(wanted is not None, f"--{field.replace('_', '-')} is required for benchmark rows")
                observed = nonempty(value.get(field), field)
                if observed != wanted:
                    selected = False
        if not selected:
            continue
        row = sanitized_query(value, number)
        key = (row["qid"], row["variant"])
        require(key not in seen, "duplicate (qid, variant) in selected queries")
        seen.add(key)
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        try:
            raw = encoded.encode("utf-8")
            line_hash = sha256(line.encode("utf-8"))
        except UnicodeError as error:
            raise ExportError("query contains an invalid Unicode scalar") from error
        rows.append(raw + b"\n")
        lineage.append({"input_line": number, "input_line_sha256": line_hash,
                        "output_line": len(rows), "output_line_sha256": sha256(raw)})
    require(rows, "no selected queries")
    manifest = {"schema": EXPORT_SCHEMA, "kind": "queries", "inputs": reader.inputs,
                "selection": {"run_id": args.run_id, "env_id": args.env_id},
                "query_fields": ["qid", "variant", "query", "options"],
                "other_top_level_fields": "excluded, including answers/gold/references/evidence",
                "query_semantics": "literal user question only; no answer instruction or gold augmentation",
                "lines": lineage}
    return {"queries.jsonl": b"".join(rows)}, manifest, reader, [Path(input_record["path"])]


def write_bundle(output, payloads, manifest, reader, protected):
    output = absolute_path(output)
    require(not any(inside(output, root) for root in protected), "output must be outside input/source roots")
    require(checked_stat(output) is None, "output directory already exists; nothing is overwritten")
    directory(output.parent)
    manifest["outputs"] = {name: {"sha256": sha256(raw), "size_bytes": len(raw)}
                           for name, raw in payloads.items()}
    payloads = {**payloads, "provenance.json": json_bytes(manifest)}
    reader.verify()
    created = []
    try:
        output.mkdir(mode=0o700)
    except OSError as error:
        raise ExportError("cannot create a new output directory; nothing is overwritten") from error
    try:
        for name, raw in payloads.items():
            path = output / name
            with path.open("xb") as stream:
                created.append(path)
                stream.write(raw)
        reader.verify()
    except BaseException:
        # Only remove files created by this invocation, never existing inputs.
        for path in reversed(created):
            path.unlink()
        output.rmdir()
        raise
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot", help="export one explicitly frozen environment/project")
    capture.add_argument("--run-dir", required=True)
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--env-id", required=True)
    capture.add_argument("--project-id", required=True)
    capture.add_argument("--code-revision", required=True)
    capture.add_argument("--capture-stage", required=True, choices=("final_state", "post_ingest", "frozen_state", "synthetic"))
    capture.add_argument("--frozen", action="store_true", help="attest that the input tree is stopped/frozen")
    authority = capture.add_mutually_exclusive_group(required=True)
    authority.add_argument("--persisted-settings-for-current-replay", action="store_true")
    authority.add_argument("--effective-memory-settings", metavar="JSON")
    capture.add_argument("--settings-provenance", help="required label for caller-supplied effective override")
    capture.add_argument("--include-scope", choices=EXTRA_SCOPES, action="append", default=[])
    capture.add_argument("--allow-source-root", action="append", default=[])
    capture.add_argument("--source-project-root", help="projectPath read root; defaults to selected frozen workspace")
    capture.add_argument("--project-root", help="original project root label only; not opened")
    capture.add_argument("--project-alias", action="append", default=[])
    capture.add_argument("--max-source-bytes", type=int, default=SOURCE_LIMIT)
    query = commands.add_parser("queries", help="independently export sanitized explicit JSONL queries")
    query.add_argument("--input", required=True)
    query.add_argument("--run-id")
    query.add_argument("--env-id")
    for command in (capture, query):
        command.add_argument("--output-dir", required=True, help="new directory; parent must already exist")
        command.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)
        command.add_argument("--max-total-bytes", type=int, default=256 * 1024 * 1024)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        payloads, manifest, reader, protected = snapshot(args) if args.command == "snapshot" else queries(args)
        output = write_bundle(args.output_dir, payloads, manifest, reader, protected)
    except (ExportError, OSError, UnicodeError, RecursionError) as error:
        print(f"export_memory_bench_snapshot: {error}", file=sys.stderr)
        return 1
    print(f"Exported {args.command} bundle to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
