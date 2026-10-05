#!/usr/bin/env python3
"""Offline compaction-quality harness (B-fix validation deliverable).

Compares two compaction configurations on the RECORDED compaction episodes and
reports a fact-survival delta. The reference dataset is the 15-day /monitor
event log (449 episodes, 141 sessions) captured in
`monitor-review-20261004-105007/dataset/compactions.jsonl`. That file is a
READ-ONLY reference: the harness never writes to it.

Why "offline": this session must not make network calls. The harness therefore
replays each recorded episode's SHAPE (tokens-before, split-turn, model, and
outcome) and synthesizes a deterministic transcript with seeded, uniquely
identifiable facts sized to that episode. A transparent, instruction-following
sumulator stands in for the summarizer: it models exactly the two policy
differences under test - the summary output budget (B3) and the prompt
instructions (B2, anchor preservation + consolidation + budget statement) - and
nothing else. Numbers from this mode measure POLICY differences, not any real
model's behavior.

Fact survival = the fraction of critical facts (anchors, decisions,
constraints) from the synthetic transcript that a compliant summarizer can
still fit inside the config's effective summary budget, prioritized by the
config's prompt instructions.

Usage:
    python scripts/validate_compaction_quality.py \
        --episodes "C:/Users/openclawuser/Optimus-Assistant/profile/workspace/monitor-review-20261004-105007/dataset/compactions.jsonl" \
        --config-a legacy \
        --config-b v2-scaled

    # custom configurations (JSON files):
    python scripts/validate_compaction_quality.py --episodes ... \
        --config-json-a '{"prompt":"v2","summary_budget_mode":"scaled"}' \
        --config-json-b '{"prompt":"legacy","summary_budget_mode":"legacy","reserve_tokens":32768}'

    # verify the budget formulas mirror the Rust implementation:
    python scripts/validate_compaction_quality.py --self-check

    # subsample for a quick pass:
    python scripts/validate_compaction_quality.py --episodes ... --max-episodes 50 --report report.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants mirrored from crates/pi-coding-agent/src/core/compaction/compaction.rs
# (cite: resolve_summary_budget_mode, summary_output_budget, build_summarization_prompt)
# ---------------------------------------------------------------------------

DEFAULT_RESERVE_TOKENS = 16_384.0
LEGACY_BUDGET_FRACTION = 0.8
SCALED_BUDGET_MIN = 2_000.0
SCALED_BUDGET_MAX = 8_000.0
SCALED_BUDGET_DIVISOR = 40.0

# Default model output ceiling used by the simulator when the episode does not
# record one (summary_output_budgets caps at model.max_tokens).
DEFAULT_MODEL_MAX_TOKENS = 8_192.0

# Prompt-instruction differences (B2). Weights model how strongly a compliant
# summarizer preserves each fact tier when the budget forces a choice.
# legacy prompt: no priority ordering, no budget statement, "never omit a
# unique required fact" (unbounded instruction, so a budget-bound model must
# improvise an ordering); v2 prompt: anchors/decisions/constraints first, then
# progress, then filler, within the stated budget.
TIER_PRIORITY_V2 = {"anchor": 0, "decision": 1, "constraint": 2, "progress": 3, "filler": 4}
TIER_PRIORITY_LEGACY = {
    "anchor": 2,
    "decision": 2,
    "constraint": 2,
    "progress": 2,
    "filler": 2,  # transcript order decides; simulated as a stable shuffle
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "legacy": {"prompt": "legacy", "summary_budget_mode": "legacy"},
    "v2": {"prompt": "v2", "summary_budget_mode": "legacy"},
    "scaled": {"prompt": "legacy", "summary_budget_mode": "scaled"},
    "v2-scaled": {"prompt": "v2", "summary_budget_mode": "scaled"},
}


@dataclass(frozen=True)
class CompactionConfig:
    """One side of the comparison. Mirrors the B1-B4 settings keys."""

    name: str
    prompt: str = "legacy"  # compaction.prompt: legacy | v2
    summary_budget_mode: str = "legacy"  # compaction.summaryBudgetMode: legacy | scaled
    reserve_tokens: float = DEFAULT_RESERVE_TOKENS  # compaction.reserveTokens
    trigger_threshold: Optional[float] = None  # compaction.triggerThreshold (B4)
    model_max_tokens: float = DEFAULT_MODEL_MAX_TOKENS

    @staticmethod
    def from_preset(name: str) -> "CompactionConfig":
        if name not in PRESETS:
            raise SystemExit(
                f"unknown preset {name!r}; choose one of {sorted(PRESETS)} or pass --config-json-a/b"
            )
        return CompactionConfig(name=name, **PRESETS[name])

    @staticmethod
    def from_json(name: str, payload: Dict[str, Any]) -> "CompactionConfig":
        known = {"prompt", "summary_budget_mode", "reserve_tokens", "trigger_threshold", "model_max_tokens"}
        unknown = set(payload) - known
        if unknown:
            raise SystemExit(f"unknown config keys in {name}: {sorted(unknown)}")
        return CompactionConfig(name=name, **payload)

    def summary_output_budget(self, tokens_before: float) -> float:
        """Mirror of summary_output_budget() in compaction.rs."""
        if self.summary_budget_mode == "scaled":
            raw = tokens_before / SCALED_BUDGET_DIVISOR
            if not (math.isfinite(raw) and raw > 0.0):
                raw = 0.0
            return math.floor(min(SCALED_BUDGET_MAX, max(SCALED_BUDGET_MIN, raw)))
        if self.summary_budget_mode != "legacy":
            raise SystemExit(f"unknown summary_budget_mode {self.summary_budget_mode!r}")
        return math.floor(LEGACY_BUDGET_FRACTION * self.reserve_tokens)

    def effective_budget(self, tokens_before: float) -> float:
        """Wire budget: min(config budget, model output ceiling) - summary_output_budgets()."""
        return min(self.summary_output_budget(tokens_before), self.model_max_tokens)

    def tier_priority(self, tier: str) -> Tuple[int, float]:
        """(priority, tiebreak noise seed). v2 orders tiers; legacy is unordered."""
        if self.prompt == "v2":
            return TIER_PRIORITY_V2[tier], 0.0
        if self.prompt != "legacy":
            raise SystemExit(f"unknown prompt version {self.prompt!r}")
        return TIER_PRIORITY_LEGACY[tier], 0.0


# ---------------------------------------------------------------------------
# Episode loading (READ-ONLY reference dataset)
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    session: str
    index: int
    started_ms: int
    outcome: str
    provider: str
    model: str
    split_turn: bool
    tokens_before: float
    duration_ms: Optional[int]

    @property
    def key(self) -> str:
        return f"{self.session}#{self.index}"


def load_episodes(path: Path, max_episodes: Optional[int] = None) -> List[Episode]:
    """Group the /monitor event log into compaction episodes.

    The dataset is an append-only event stream: `compaction` rows open an
    episode; `compaction_prefix` rows mark split-turn episodes; terminal rows
    (`success` / `failure` / `cancelled`) close them. `serialized_bytes` on the
    `compaction_prepare` terminal row measures the summarized payload, which is
    the closest recorded proxy for `tokens_before`.
    """
    events: Dict[Tuple[str, int], Dict[str, Any]] = {}
    order: List[Tuple[str, int]] = []
    counters: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            session = record.get("session")
            op = record.get("op")
            if session is None or op is None:
                continue
            if op == "compaction" and record.get("outcome") == "started":
                index = counters.get(session, 0)
                counters[session] = index + 1
                key = (session, index)
                events[key] = {
                    "started_ms": record.get("ts"),
                    "model": record.get("model"),
                    "provider": record.get("provider"),
                    "outcome": "started",
                    "duration_ms": None,
                    "tokens_before": None,
                    "split_turn": False,
                }
                order.append(key)
                continue
            if op == "compaction_prefix" and record.get("outcome") == "started":
                # The nearest open episode for this session is a split turn.
                index = counters.get(session, 1) - 1
                entry = events.get((session, index))
                if entry is not None:
                    entry["split_turn"] = True
                continue
            if op in (
                "compaction",
                "compaction_prepare",
                "compaction_restore",
                "compaction_history",
                "compaction_prefix",
            ):
                index = counters.get(session, 1) - 1
                entry = events.get((session, index))
                if entry is None:
                    continue
                outcome = record.get("outcome")
                if outcome in ("success", "failure", "cancelled"):
                    entry["outcome"] = outcome
                metrics = record.get("m") or {}
                if op in ("compaction_history", "compaction_prefix") and outcome != "started":
                    # The summarized payload size: history (plus prefix for a
                    # split turn) is what the summarizer must fit.
                    if "serialized_bytes" in metrics:
                        current = entry.get("serialized_bytes") or 0
                        entry["serialized_bytes"] = current + metrics["serialized_bytes"]
                if op == "compaction" and outcome != "started":
                    if "total_ms" in metrics:
                        entry["duration_ms"] = metrics["total_ms"]
                continue

    episodes: List[Episode] = []
    for key in order:
        entry = events[key]
        serialized = entry.get("serialized_bytes")
        # serialized_bytes is a UTF-8 byte count; ~4 bytes per token is the
        # same estimate the Rust summarizer uses for sizing decisions.
        tokens_before = (serialized / 4.0) if isinstance(serialized, (int, float)) else 20_000.0
        episodes.append(
            Episode(
                session=key[0],
                index=key[1],
                started_ms=entry.get("started_ms") or 0,
                outcome=str(entry["outcome"]),
                provider=str(entry.get("provider") or "unknown"),
                model=str(entry.get("model") or "unknown"),
                split_turn=bool(entry["split_turn"]),
                tokens_before=float(tokens_before),
                duration_ms=entry.get("duration_ms"),
            )
        )
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


# ---------------------------------------------------------------------------
# Synthetic transcript with uniquely identifiable facts
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    tier: str  # anchor | decision | constraint | progress | filler
    ident: str
    tokens: float


@dataclass
class Transcript:
    episode_key: str
    tokens_before: float
    facts: List[Fact] = field(default_factory=list)

    def critical_facts(self) -> List[Fact]:
        return [fact for fact in self.facts if fact.tier in ("anchor", "decision", "constraint")]


def synthesize_transcript(episode: Episode, seed: int) -> Transcript:
    """Deterministic per-episode transcript sized to the recorded token count.

    Fact mix follows the structured summary sections: every ~40 tokens of input
    (the B3 divisor) carries one fact; 25% of facts are critical tiers.
    """
    rng = random.Random(f"{seed}:{episode.key}")
    target_facts = max(8, int(episode.tokens_before / SCALED_BUDGET_DIVISOR))
    facts: List[Fact] = []
    for i in range(target_facts):
        roll = rng.random()
        if roll < 0.08:
            tier = "anchor"
        elif roll < 0.18:
            tier = "decision"
        elif roll < 0.25:
            tier = "constraint"
        elif roll < 0.65:
            tier = "progress"
        else:
            tier = "filler"
        facts.append(Fact(tier=tier, ident=f"{episode.key}:{tier}:{i}", tokens=40.0))
    # A split-turn episode also carries the turn-prefix facts (kept separately
    # by the real implementation, so both configs see them).
    if episode.split_turn:
        for i in range(4):
            facts.append(Fact(tier="anchor", ident=f"{episode.key}:prefix:{i}", tokens=40.0))
    return Transcript(episode_key=episode.key, tokens_before=episode.tokens_before, facts=facts)


# ---------------------------------------------------------------------------
# The compliant-summarizer simulator
# ---------------------------------------------------------------------------

def simulate_summary(transcript: Transcript, config: CompactionConfig, seed: int) -> List[Fact]:
    """Which facts a compliant summarizer keeps under this config.

    The simulator encodes ONLY the two policy differences under test:
      1. the effective wire budget (B3 mode + model ceiling), and
      2. the prompt's prioritization (B2: v2 orders critical tiers first and
         states the budget; legacy gives no ordering and no budget statement).
    Everything else (model quality, consolidation skill) is identical by
    construction, so any survival delta is attributable to policy.
    """
    rng = random.Random(f"{seed}:{config.name}:{transcript.episode_key}")
    budget = config.effective_budget(transcript.tokens_before)
    priority = {tier: config.tier_priority(tier) for tier in TIER_PRIORITY_V2}
    if config.prompt == "legacy":
        # No stated ordering: a budget-bound model keeps facts in an
        # effectively arbitrary (here: seeded-shuffled) order.
        ordered = list(transcript.facts)
        rng.shuffle(ordered)
    else:
        ordered = sorted(
            transcript.facts,
            key=lambda fact: (priority[fact.tier][0], transcript.facts.index(fact)),
        )
    kept: List[Fact] = []
    used = 0.0
    for fact in ordered:
        if used + fact.tokens <= budget:
            kept.append(fact)
            used += fact.tokens
    return kept


def survival_rate(kept: Sequence[Fact], critical: Sequence[Fact]) -> float:
    if not critical:
        return 1.0
    kept_ids = {fact.ident for fact in kept}
    return sum(1 for fact in critical if fact.ident in kept_ids) / len(critical)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class SideResult:
    config: CompactionConfig
    episodes: int = 0
    survival_sum: float = 0.0
    survival_min: float = 1.0
    survival_max: float = 0.0
    budget_sum: float = 0.0
    anchor_survival_sum: float = 0.0
    anchor_episodes: int = 0

    def observe(self, survival: float, budget: float, anchor_survival: Optional[float]) -> None:
        self.episodes += 1
        self.survival_sum += survival
        self.survival_min = min(self.survival_min, survival)
        self.survival_max = max(self.survival_max, survival)
        self.budget_sum += budget
        if anchor_survival is not None:
            self.anchor_survival_sum += anchor_survival
            self.anchor_episodes += 1

    @property
    def mean_survival(self) -> float:
        return self.survival_sum / self.episodes if self.episodes else 0.0

    @property
    def mean_budget(self) -> float:
        return self.budget_sum / self.episodes if self.episodes else 0.0

    @property
    def mean_anchor_survival(self) -> float:
        return self.anchor_survival_sum / self.anchor_episodes if self.anchor_episodes else 0.0


def compare(
    episodes: Sequence[Episode],
    config_a: CompactionConfig,
    config_b: CompactionConfig,
    seed: int,
) -> Dict[str, Any]:
    side_a = SideResult(config_a)
    side_b = SideResult(config_b)
    per_episode: List[Dict[str, Any]] = []

    for episode in episodes:
        transcript = synthesize_transcript(episode, seed)
        critical = transcript.critical_facts()
        anchors = [fact for fact in transcript.facts if fact.tier == "anchor"]

        kept_a = simulate_summary(transcript, config_a, seed)
        kept_b = simulate_summary(transcript, config_b, seed)
        survival_a = survival_rate(kept_a, critical)
        survival_b = survival_rate(kept_b, critical)
        budget_a = config_a.effective_budget(transcript.tokens_before)
        budget_b = config_b.effective_budget(transcript.tokens_before)
        anchor_a = survival_rate(kept_a, anchors) if anchors else None
        anchor_b = survival_rate(kept_b, anchors) if anchors else None

        side_a.observe(survival_a, budget_a, anchor_a)
        side_b.observe(survival_b, budget_b, anchor_b)
        per_episode.append(
            {
                "episode": episode.key,
                "session": episode.session,
                "outcome": episode.outcome,
                "model": episode.model,
                "split_turn": episode.split_turn,
                "tokens_before": round(episode.tokens_before, 1),
                "budget_a": budget_a,
                "budget_b": budget_b,
                "survival_a": round(survival_a, 4),
                "survival_b": round(survival_b, 4),
                "delta": round(survival_b - survival_a, 4),
            }
        )

    return {
        "config_a": vars(config_a) | {"name": config_a.name},
        "config_b": vars(config_b) | {"name": config_b.name},
        "episodes": len(episodes),
        "summary": {
            "a": {
                "mean_critical_fact_survival": round(side_a.mean_survival, 4),
                "min": round(side_a.survival_min, 4) if side_a.episodes else None,
                "max": round(side_a.survival_max, 4) if side_a.episodes else None,
                "mean_anchor_survival": round(side_a.mean_anchor_survival, 4),
                "mean_effective_budget_tokens": round(side_a.mean_budget, 1),
            },
            "b": {
                "mean_critical_fact_survival": round(side_b.mean_survival, 4),
                "min": round(side_b.survival_min, 4) if side_b.episodes else None,
                "max": round(side_b.survival_max, 4) if side_b.episodes else None,
                "mean_anchor_survival": round(side_b.mean_anchor_survival, 4),
                "mean_effective_budget_tokens": round(side_b.mean_budget, 1),
            },
            "delta_mean_critical_fact_survival": round(side_b.mean_survival - side_a.mean_survival, 4),
            "delta_mean_anchor_survival": round(
                side_b.mean_anchor_survival - side_a.mean_anchor_survival, 4
            ),
        },
        "per_episode": per_episode,
    }


# ---------------------------------------------------------------------------
# Self-check: the budget formulas must mirror the Rust implementation
# ---------------------------------------------------------------------------

def self_check() -> None:
    legacy = CompactionConfig(name="check-legacy")
    scaled = CompactionConfig(name="check-scaled", summary_budget_mode="scaled")

    assert legacy.summary_output_budget(250_905.0) == math.floor(0.8 * DEFAULT_RESERVE_TOKENS)
    assert legacy.summary_output_budget(1_000_000.0) == 13_107.0
    assert scaled.summary_output_budget(250_905.0) == 6_272.0
    assert scaled.summary_output_budget(40_000.0) == 2_000.0
    assert scaled.summary_output_budget(20_000.0) == 2_000.0
    assert scaled.summary_output_budget(400_000.0) == 8_000.0
    assert scaled.summary_output_budget(0.0) == 2_000.0
    assert scaled.summary_output_budget(float("nan")) == 2_000.0
    # The wire budget is additionally capped by the model ceiling.
    tight = CompactionConfig(name="tight", summary_budget_mode="scaled", model_max_tokens=1_000.0)
    assert tight.effective_budget(400_000.0) == 1_000.0
    print("self-check: budget formulas match compaction.rs (legacy floor(0.8*reserve); scaled clamp(t/40, 2000, 8000))")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(preset: Optional[str], json_spec: Optional[str], fallback_name: str) -> CompactionConfig:
    if preset and json_spec:
        raise SystemExit(f"--{fallback_name}: pass either a preset or JSON, not both")
    if preset:
        return CompactionConfig.from_preset(preset)
    if json_spec:
        return CompactionConfig.from_json(fallback_name, json.loads(json_spec))
    raise SystemExit(f"--{fallback_name} is required (preset or JSON)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, default=None, help="recorded compactions.jsonl (READ-ONLY)")
    parser.add_argument("--config-a", default=None, help=f"preset: {sorted(PRESETS)}")
    parser.add_argument("--config-b", default=None, help=f"preset: {sorted(PRESETS)}")
    parser.add_argument("--config-json-a", default=None, help="config A as inline JSON")
    parser.add_argument("--config-json-b", default=None, help="config B as inline JSON")
    parser.add_argument("--max-episodes", type=int, default=None, help="subsample the first N episodes")
    parser.add_argument("--seed", type=int, default=20261004, help="deterministic synthesis seed")
    parser.add_argument("--report", type=Path, default=None, help="write the full JSON report here")
    parser.add_argument("--self-check", action="store_true", help="verify budget formulas against the Rust constants")
    args = parser.parse_args(argv)

    if args.self_check:
        self_check()
        if args.episodes is None:
            return 0

    if args.episodes is None:
        parser.error("--episodes is required unless --self-check is used alone")
    if not args.episodes.exists():
        raise SystemExit(f"episodes file not found: {args.episodes}")

    config_a = build_config(args.config_a, args.config_json_a, "config-a")
    config_b = build_config(args.config_b, args.config_json_b, "config-b")

    episodes = load_episodes(args.episodes, args.max_episodes)
    if not episodes:
        raise SystemExit("no compaction episodes found in the dataset")

    report = compare(episodes, config_a, config_b, args.seed)

    print(f"episodes: {report['episodes']}")
    print(f"config A: {config_a.name} -> {report['summary']['a']}")
    print(f"config B: {config_b.name} -> {report['summary']['b']}")
    print(f"delta (B - A): critical-fact survival {report['summary']['delta_mean_critical_fact_survival']:+.4f}, "
          f"anchor survival {report['summary']['delta_mean_anchor_survival']:+.4f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"report written: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
