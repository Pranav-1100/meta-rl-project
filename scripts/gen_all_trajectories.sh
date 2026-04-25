#!/usr/bin/env bash
#
# Generate the full ~320-episode synthetic trajectory mix for SFT warmup.
#
# Mix matches CONTEXT.md §23.2:
#   easy_ria_late                     80
#   medium_jay_standup                60
#   hard_dinner_sushi                 40
#   complex_multi_objective_dinner    20
#   recovery_mom_missed_call          20
#   honest_failure_hibachi            30
#   multi_day_reschedule              30
#   group_order_lunch_budget          20
#   composite_ria_late_and_dinner     20
#                                  -----
#                                    320
#
# Held-out adversarial tasks (adv_*) are intentionally NOT generated — they're
# reserved for evaluating whether the trained model learned to fail honestly.
#
# Requires ANTHROPIC_API_KEY in .env. Resumes by appending to existing JSONLs,
# so re-running won't lose data — but be aware seeds will overlap if --seed-start
# isn't bumped. For a clean re-run: rm data/trajectories/*.jsonl first.
#
# Usage:
#   bash scripts/gen_all_trajectories.sh 2>&1 | tee data/gen.log

set -e

cd "$(dirname "$0")/.."
mkdir -p data/trajectories

echo "===> $(date) starting trajectory generation"
uv run python scripts/gen_trajectories.py --task easy_ria_late                  --count 80 --seed-start 1
uv run python scripts/gen_trajectories.py --task medium_jay_standup             --count 60 --seed-start 1
uv run python scripts/gen_trajectories.py --task hard_dinner_sushi              --count 40 --seed-start 1
uv run python scripts/gen_trajectories.py --task complex_multi_objective_dinner --count 20 --seed-start 1
uv run python scripts/gen_trajectories.py --task recovery_mom_missed_call       --count 20 --seed-start 1
uv run python scripts/gen_trajectories.py --task honest_failure_hibachi         --count 30 --seed-start 1
uv run python scripts/gen_trajectories.py --task multi_day_reschedule           --count 30 --seed-start 1
uv run python scripts/gen_trajectories.py --task group_order_lunch_budget       --count 20 --seed-start 1
uv run python scripts/gen_trajectories.py --task composite_ria_late_and_dinner  --count 20 --seed-start 1

echo
echo "===> $(date) done. JSONL line counts:"
wc -l data/trajectories/*.jsonl
