#!/usr/bin/env python3
"""Plot SFT (and GRPO if present) loss curves from HF Jobs stdout logs.

The training script prints standard ``transformers.Trainer`` loss lines like::

    {'loss': 1.5821, 'grad_norm': 2.1, 'learning_rate': 1.8e-05, 'epoch': 0.07}

We grep for those lines, parse, and plot. Required as 'evidence of training'
for the OpenEnv hackathon submission.

Usage::

    # 1) save the training-job stdout to a local file:
    hf jobs logs <qwen_training_job_id>  > /tmp/qwen_training.log
    hf jobs logs <gemma_training_job_id> > /tmp/gemma_training.log

    # 2) plot from those files:
    python scripts/plot_training_loss.py \
        --qwen-log /tmp/qwen_training.log \
        --gemma-log /tmp/gemma_training.log

    # plots → data/plots/sft_loss.png
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOSS_RE = re.compile(r"\{[^{}]*'loss'\s*:\s*[0-9.eE+-]+[^{}]*\}")


def parse_log(path: Path) -> list[dict]:
    """Return list of {'loss','epoch',...} dicts in the order they appear."""
    if not path.exists():
        print(f"  ✗ {path} not found")
        return []
    text = path.read_text(errors="ignore")
    rows: list[dict] = []
    for m in LOSS_RE.finditer(text):
        try:
            d = ast.literal_eval(m.group(0))
            if isinstance(d, dict) and "loss" in d:
                rows.append(d)
        except (ValueError, SyntaxError):
            continue
    return rows


def plot_panel(ax, label: str, rows: list[dict]) -> bool:
    if not rows:
        ax.text(0.5, 0.5, f"no loss lines in log\n{label}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888")
        ax.set_title(f"{label}: log missing")
        return False

    losses = [r["loss"] for r in rows]
    if any("epoch" in r for r in rows):
        x = [r.get("epoch", i) for i, r in enumerate(rows)]
        ax.set_xlabel("epoch")
    else:
        x = list(range(len(rows)))
        ax.set_xlabel("step")
    ax.plot(x, losses, color="#2563eb", linewidth=1.6, alpha=0.85)
    ax.scatter(x, losses, s=14, color="#2563eb", alpha=0.55)
    ax.set_ylabel("loss")
    ax.set_title(f"{label} — SFT loss "
                 f"({len(losses)} pts: {losses[0]:.3f} → {losses[-1]:.3f})")
    ax.grid(alpha=0.3)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-log", type=Path, default=Path("/tmp/qwen_training.log"))
    p.add_argument("--gemma-log", type=Path, default=Path("/tmp/gemma_training.log"))
    p.add_argument("--out", type=Path, default=OUT_DIR / "sft_loss.png")
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    panels = [
        (axes[0], "Qwen 2.5 7B",  args.qwen_log),
        (axes[1], "Gemma 2 9B",   args.gemma_log),
    ]
    found_any = False
    for ax, label, path in panels:
        print(f"[parse] {label}: {path}")
        rows = parse_log(path)
        ok = plot_panel(ax, label, rows)
        found_any = found_any or ok
        if ok:
            print(f"  ✓ {len(rows)} loss points: "
                  f"{rows[0]['loss']:.3f} → {rows[-1]['loss']:.3f}")

    fig.suptitle("PhonePilot — SFT Training Loss (evidence of training)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\n[plot] saved → {args.out}")
    return 0 if found_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
