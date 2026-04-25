#!/usr/bin/env python3
"""Run the 10 capability probes against a policy and emit JSON + a curve plot.

Each probe is a tiny single-skill task ("send a one-line WhatsApp", "find a pizza on
Zomato", etc.). A passing rate of 8–10 / 10 is roughly what a model needs to be
reliable enough for the harder composite tasks. Run as a battery every N training steps
and plot ``probes_passed_out_of_10`` over time for a clean monotonic learning curve.

Outputs
-------

* ``data/eval/probes_<policy>.json`` — full per-probe result + summary count.
* If ``--checkpoint-tag`` is passed, ``data/eval/probes_<policy>_<tag>.json``.
* If multiple snapshots exist, ``data/plots/probes_curve.png`` shows the trajectory.

Run with::

    uv run python scripts/run_probes.py --policy scripted_easy
    uv run python scripts/run_probes.py --policy random --checkpoint-tag step_0
    uv run python scripts/run_probes.py --policy trained \
        --model-path ./models/grpo_lora --checkpoint-tag step_120
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phonepilot_env.env import build_env  # noqa: E402
from phonepilot_env.probes import PROBES, run_probes_with_policy  # noqa: E402

from run_episode import POLICIES  # type: ignore[import-not-found]  # noqa: E402

EVAL_DIR = REPO_ROOT / "data" / "eval"
PLOTS_DIR = REPO_ROOT / "data" / "plots"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_model_policy(model_path: str, label: str):
    """Lazy-import the heavy stack and return a probes-compatible policy."""
    from eval import load_model_policy  # type: ignore[import-not-found]

    return load_model_policy(model_path, label)


def run(policy_name: str, model_path: str | None) -> dict:
    if model_path is not None:
        policy = _load_model_policy(model_path, policy_name)
    elif policy_name in POLICIES:
        policy = POLICIES[policy_name]
    else:
        raise SystemExit(
            f"Unknown policy {policy_name!r}. Built-in: {sorted(POLICIES)}. "
            "Or pass --model-path for a trained-model policy."
        )
    results = run_probes_with_policy(build_env, policy)
    n_passed = sum(1 for v in results.values() if v)
    return {
        "policy": policy_name,
        "n_passed": n_passed,
        "n_total": len(PROBES),
        "score": n_passed / max(1, len(PROBES)),
        "by_probe": {k: bool(v) for k, v in results.items()},
    }


def _plot_curve(policy: str) -> None:
    """If multiple checkpointed JSONs exist for this policy, plot the trajectory."""
    import matplotlib.pyplot as plt

    pat = re.compile(rf"^probes_{re.escape(policy)}_step_(\d+)\.json$")
    points: list[tuple[int, int]] = []
    for f in EVAL_DIR.glob(f"probes_{policy}_*.json"):
        m = pat.match(f.name)
        if not m:
            continue
        step = int(m.group(1))
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        points.append((step, int(d.get("n_passed", 0))))
    if len(points) < 2:
        return  # need at least 2 points for a curve
    points.sort()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", linewidth=2.5, color="#2ecc71")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Probes passed (out of 10)")
    ax.set_ylim(-0.5, 10.5)
    ax.set_yticks(range(0, 11))
    ax.grid(alpha=0.25)
    ax.set_title(f"Capability probes over training — {policy}")
    fig.tight_layout()
    out = PLOTS_DIR / "probes_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({len(points)} checkpoints)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy",
        default="scripted_easy",
        help="Built-in policy name (random/null/scripted_easy) OR an arbitrary label "
        "for a trained model when paired with --model-path.",
    )
    p.add_argument("--model-path", default=None, help="Local path to a HF model dir for the trained-policy case.")
    p.add_argument(
        "--checkpoint-tag",
        default=None,
        help="Optional tag (e.g. step_120) — namespaces the output file so multiple "
        "snapshots can be plotted as a curve.",
    )
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    result = run(args.policy, args.model_path)

    suffix = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
    out_path = EVAL_DIR / f"probes_{args.policy}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"{result['policy']:<24} passed {result['n_passed']}/{result['n_total']}  "
        f"({result['score']:.0%}) → {out_path.name}"
    )
    failed = [k for k, v in result["by_probe"].items() if not v]
    if failed:
        print(f"  failed probes: {', '.join(failed)}")

    if not args.no_plot:
        _plot_curve(args.policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
