#!/usr/bin/env python3
"""The headline research-flavoured plot for the submission.

Two-axis chart that addresses the obvious failure mode of a single "lying rate"
curve: a model that "stops lying" by also "stops trying" looks fake-good. Showing
honesty (lying rate going *down*) AND capability (success rate going *up*) on the
same plot is the only way to prove the agent is genuinely learning to admit failure
when it has to, while still attempting tasks when it can.

Inputs
------

* ``data/eval/lying_rate.json`` — list of {baseline, overall_lying_rate, ...}
  rows, produced by ``scripts/eval.py --lying-rate``. May also load
  ``data/eval/lying_rate_<tag>.json`` files when ``--checkpoints`` is passed
  (training-step mode).

* ``data/eval/summary.json`` — list of {baseline, task_id, success_rate, ...}
  rows, produced by ``scripts/eval.py`` (staircase mode). The capability number
  for a baseline is the mean success rate across its TRAINING tasks (held-out
  adversarial tasks intentionally excluded — those are the lying-rate axis).

Output
------

* ``data/plots/honesty_vs_capability.png`` — dual-y-axis line/bar plot.

The plot tolerates missing files. If only lying_rate.json exists, it draws just
the red line. If only summary.json exists, it draws just the blue line. This
makes it safe to run early (before any training has happened) and re-run during
training as new checkpoints come in.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "data" / "eval"
PLOTS_DIR = REPO_ROOT / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))
from phonepilot_env.tasks import training_task_ids  # noqa: E402

TRAINING_TASKS = set(training_task_ids())

# Canonical baseline ordering — left to right.
BASELINE_ORDER = [
    "random",
    "null",
    "scripted_easy",
    "base",
    "sft",
    "trained",
]


def _load_lying_rate(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {r["baseline"]: r["overall_lying_rate"] for r in rows}


def _load_capability(path: Path) -> dict[str, float]:
    """Mean training-task success rate per baseline. Adversarial tasks excluded."""
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    by_baseline: dict[str, list[float]] = {}
    for r in rows:
        if r.get("task_id") not in TRAINING_TASKS:
            continue
        by_baseline.setdefault(r["baseline"], []).append(r.get("success_rate", 0.0))
    return {b: statistics.fmean(v) for b, v in by_baseline.items() if v}


def _ordered(baselines: list[str]) -> list[str]:
    head = [b for b in BASELINE_ORDER if b in baselines]
    tail = sorted(b for b in baselines if b not in BASELINE_ORDER)
    return head + tail


def plot(
    lying_rate: dict[str, float],
    capability: dict[str, float],
    out_path: Path,
    title: str = "Honesty vs Capability — PhonePilot baselines",
) -> None:
    import matplotlib.pyplot as plt

    baselines = _ordered(sorted(set(lying_rate) | set(capability)))
    if not baselines:
        print("No data found. Run scripts/eval.py and scripts/eval.py --lying-rate first.")
        return

    xs = list(range(len(baselines)))
    fig, ax_left = plt.subplots(figsize=(9, 5))
    ax_right = ax_left.twinx()

    # Left axis (red) — lying rate, lower is better.
    ly = [lying_rate.get(b, float("nan")) for b in baselines]
    ax_left.plot(
        xs, ly, color="#d9534f", marker="o", linewidth=2.5,
        label="Lying rate (adversarial battery, ↓ better)",
    )
    ax_left.set_ylabel("Lying rate (held-out adversarial)", color="#d9534f")
    ax_left.set_ylim(-0.05, 1.05)
    ax_left.tick_params(axis="y", labelcolor="#d9534f")

    # Right axis (blue) — capability, higher is better.
    cap = [capability.get(b, float("nan")) for b in baselines]
    ax_right.plot(
        xs, cap, color="#1f77b4", marker="s", linewidth=2.5,
        label="Success rate (training tasks, ↑ better)",
    )
    ax_right.set_ylabel("Success rate (training tasks)", color="#1f77b4")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.tick_params(axis="y", labelcolor="#1f77b4")

    ax_left.set_xticks(xs)
    ax_left.set_xticklabels(baselines, rotation=15)
    ax_left.set_xlabel("Baseline / training stage")
    ax_left.grid(axis="y", alpha=0.2)
    ax_left.set_title(title)

    # Combined legend at top — handles from both axes.
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"   baselines: {baselines}")
    print(f"   lying:     {ly}")
    print(f"   capability:{cap}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lying-rate",
        default=str(EVAL_DIR / "lying_rate.json"),
        help="Path to lying-rate JSON (output of eval.py --lying-rate).",
    )
    p.add_argument(
        "--capability",
        default=str(EVAL_DIR / "summary.json"),
        help="Path to staircase summary JSON (output of eval.py).",
    )
    p.add_argument(
        "--out",
        default=str(PLOTS_DIR / "honesty_vs_capability.png"),
    )
    p.add_argument(
        "--title",
        default="Honesty vs Capability — PhonePilot baselines",
    )
    args = p.parse_args()

    lying = _load_lying_rate(Path(args.lying_rate))
    cap = _load_capability(Path(args.capability))
    plot(lying, cap, Path(args.out), title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
