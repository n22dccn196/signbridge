#!/usr/bin/env python
"""
Live monitor for robust WLASL pipeline.

Tracks:
- Model-level completion (tuned/trained)
- Phase-level completion (optimize/eval/full done)
- Current tuning trial progress from pipeline log

Writes concise events to stdout and to a monitor log file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


TUNE_HEADER_RE = re.compile(r"TUNING:\s+([A-Z0-9_]+)\s+\((\d+)\s+trials", re.IGNORECASE)
TRIAL_RE = re.compile(r"Trial\s+(\d+)/(\d+)\s+val=", re.IGNORECASE)


@dataclass
class MonitorState:
    tuned: Dict[str, str] = field(default_factory=dict)
    trained: Dict[str, str] = field(default_factory=dict)
    optimized: bool = False
    evaluated: bool = False
    completed_at: Optional[str] = None


@dataclass
class TrialProgress:
    model: Optional[str] = None
    done: int = 0
    total: int = 0


def now_s() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def emit(msg: str, out_file: Path) -> None:
    line = f"[{now_s()}] {msg}"
    print(line, flush=True)
    append_line(out_file, line)


def parse_trial_progress(log_file: Path) -> TrialProgress:
    if not log_file.exists():
        return TrialProgress()

    current_model: Optional[str] = None
    current_done = 0
    current_total = 0

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = TUNE_HEADER_RE.search(line)
            if m:
                current_model = m.group(1).lower()
                current_done = 0
                current_total = int(m.group(2))
                continue

            t = TRIAL_RE.search(line)
            if t:
                current_done = int(t.group(1))
                current_total = int(t.group(2))

    return TrialProgress(model=current_model, done=current_done, total=current_total)


def diff_dict_events(kind: str, prev: Dict[str, str], cur: Dict[str, str]) -> Tuple[list[str], Dict[str, str]]:
    events: list[str] = []
    merged = dict(prev)
    for model, st in cur.items():
        old = prev.get(model)
        if old != st:
            events.append(f"{kind} model={model} status={st}")
        merged[model] = st
    return events, merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Live monitor for robust WLASL orchestrator")
    parser.add_argument("--variant", default="wlasl2000", choices=["wlasl100", "wlasl2000"])
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--stop-when-complete", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    runs_dir = script_dir / ("runs_wlasl2000" if args.variant == "wlasl2000" else "runs_wlasl")
    state_file = runs_dir / f"robust_state_{args.variant}.json"
    pipe_log = runs_dir / f"robust_pipeline_{args.variant}.log"
    monitor_log = runs_dir / f"robust_monitor_{args.variant}.log"

    emit(
        f"monitor_start variant={args.variant} poll={args.poll_seconds}s state={state_file.name} log={pipe_log.name}",
        monitor_log,
    )

    snapshot = MonitorState()
    last_trial = TrialProgress()
    last_heartbeat_ts = 0.0

    while True:
        data = read_json(state_file)
        cur = MonitorState(
            tuned=data.get("tuned", {}),
            trained=data.get("trained", {}),
            optimized=bool(data.get("optimized", False)),
            evaluated=bool(data.get("evaluated", False)),
            completed_at=data.get("completed_at"),
        )

        tuned_events, tuned_snapshot = diff_dict_events("tune", snapshot.tuned, cur.tuned)
        trained_events, trained_snapshot = diff_dict_events("train", snapshot.trained, cur.trained)
        for e in tuned_events + trained_events:
            emit(e, monitor_log)

        snapshot.tuned = tuned_snapshot
        snapshot.trained = trained_snapshot

        if (not snapshot.optimized) and cur.optimized:
            emit("phase_complete optimize=true", monitor_log)
        if (not snapshot.evaluated) and cur.evaluated:
            emit("phase_complete evaluate=true", monitor_log)
        if (snapshot.completed_at is None) and cur.completed_at:
            emit(f"pipeline_complete completed_at={cur.completed_at}", monitor_log)

        snapshot.optimized = cur.optimized
        snapshot.evaluated = cur.evaluated
        snapshot.completed_at = cur.completed_at

        trial = parse_trial_progress(pipe_log)
        if trial.model and (trial.model != last_trial.model or trial.done != last_trial.done or trial.total != last_trial.total):
            emit(f"tune_progress model={trial.model} trial={trial.done}/{trial.total}", monitor_log)
            last_trial = trial

        now = time.time()
        if now - last_heartbeat_ts > max(args.poll_seconds * 3, 60):
            emit(
                (
                    f"heartbeat tuned={sum(1 for v in snapshot.tuned.values() if v == 'ok')} "
                    f"trained={sum(1 for v in snapshot.trained.values() if v == 'ok')} "
                    f"optimized={snapshot.optimized} evaluated={snapshot.evaluated}"
                ),
                monitor_log,
            )
            last_heartbeat_ts = now

        if args.stop_when_complete and snapshot.completed_at:
            emit("monitor_stop reason=pipeline_complete", monitor_log)
            return 0

        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    raise SystemExit(main())
