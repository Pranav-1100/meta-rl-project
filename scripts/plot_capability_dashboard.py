#!/usr/bin/env python3
"""Six-panel capability-dashboard plot.

Reads ``data/dashboard.csv`` and produces a 2×3 subplot grid where each panel is
one of the dashboard metrics over training steps. This is the "showing
improvement" hedge: even when aggregate reward is noisy, 3-4 of these panels
should trend cleanly, giving us monotonic-ish curves to point at.

CSV schema (the training notebook is expected to append one row per rollout):

    step,channel_appropriateness,spam_rate,time_appropriate_rate,truthfulness,efficiency,recovery_rate
    0,0.3,0.6,0.7,0.5,0.05,1.0
    5,0.4,0.55,0.7,0.5,0.08,1.0
    ...

The plot tolerates:
  * the file being missing entirely (warns, exits 0)
  * fewer rows than panels expect (just plots whatever's there)
  * extra columns (ignored)

A small EMA smoothing is applied per panel so the curves read cleanly without
hiding the underlying signal.

Run with:

    uv run python scripts/plot_capability_dashboard.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_CSV = REPO_ROOT / "data" / "dashboard.csv"
PLOTS_DIR = REPO_ROOT / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [
    ("channel_appropriateness", "Channel appropriateness", True),   # higher better
    ("spam_rate",                "Spam rate",                False),  # lower better
    ("time_appropriate_rate",   "Time-of-day appropriateness", True),
    ("truthfulness",             "Truthfulness",             True),
    ("efficiency",               "Efficiency (sub-goals/action)", True),
    ("recovery_rate",            "Recovery rate",            True),
]


def _read_csv(path: Path) -> tuple[list[int], dict[str, list[float]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], {}
    steps: list[int] = []
    cols: dict[str, list[float]] = {name: [] for name, _, _ in PANELS}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(float(row["step"])))
            except (KeyError, ValueError):
                continue
            for name, _, _ in PANELS:
                try:
                    cols[name].append(float(row[name]))
                except (KeyError, ValueError):
                    cols[name].append(float("nan"))
    return steps, cols


def _ema(xs: list[float], alpha: float = 0.3) -> list[float]:
    out: list[float] = []
    s: float | None = None
    for x in xs:
        if x != x:  # NaN
            out.append(float("nan"))
            continue
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return out


def plot(csv_path: Path, out_path: Path, title: str | None = None) -> int:
    import matplotlib.pyplot as plt

    steps, cols = _read_csv(csv_path)
    if not steps:
        print(
            f"No dashboard data at {csv_path} yet. The training notebook should "
            "append rows during GRPO. Re-run this script after training to "
            "produce the plot."
        )
        # Still emit an empty placeholder so downstream tooling doesn't crash.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5,
            "No dashboard data yet.\n\nTraining notebook will populate\n`data/dashboard.csv`.",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
        )
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote placeholder {out_path}")
        return 0

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (name, label, higher_better) in zip(axes.flat, PANELS):
        raw = cols.get(name, [])
        smooth = _ema(raw)
        ax.plot(steps, raw, color="#cccccc", linewidth=1.0, label="raw")
        ax.plot(steps, smooth, color="#1f77b4", linewidth=2.0, label="EMA(0.3)")
        ax.set_title(label + (" ↑" if higher_better else " ↓"))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Metric value")
    fig.suptitle(title or "PhonePilot — capability dashboard over training", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(steps)} steps logged)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DASHBOARD_CSV))
    p.add_argument("--out", default=str(PLOTS_DIR / "capability_dashboard.png"))
    p.add_argument("--title", default=None)
    args = p.parse_args()
    return plot(Path(args.csv), Path(args.out), title=args.title)


if __name__ == "__main__":
    raise SystemExit(main())
