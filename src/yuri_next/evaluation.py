"""Repeatable paired game runs and a compact match report."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def wilson_interval(wins: int, games: int, z: float = 1.96):
    if games <= 0:
        return None
    proportion = wins / games
    denominator = 1 + z * z / games
    center = (proportion + z * z / (2 * games)) / denominator
    radius = z * ((proportion * (1 - proportion) / games + z * z / (4 * games * games)) ** 0.5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def run_paired_evaluation(*, policies, games_per_policy, seed_start, map_name,
                          difficulty, realtime, output):
    if len(policies) < 2 or len(set(policies)) != len(policies):
        raise ValueError("Choose at least two distinct policies for a paired evaluation")
    if games_per_policy <= 0 or seed_start < 0:
        raise ValueError("games_per_policy must be positive and seed_start nonnegative")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path("artifacts/games")
    runs = []
    pairs = []
    for seed in range(seed_start, seed_start + games_per_policy):
        pair = {"seed": seed, "runs": []}
        for policy in policies:
            run_id = f"eval-{run_stamp}-{policy}-seed{seed}"
            trace_path = run_root / f"{run_id}-{policy}.jsonl"
            metadata_path = trace_path.with_suffix(".meta.json")
            command = [sys.executable, "-m", "yuri_next.cli", "game",
                       "--policy", policy, "--map", map_name,
                       "--difficulty", difficulty, "--seed", str(seed),
                       "--run-id", run_id]
            if realtime:
                command.append("--realtime")
            process = subprocess.run(command, check=False)
            row = {"policy": policy, "seed": seed, "run_id": run_id,
                   "process_exit_code": process.returncode,
                   "metadata_path": str(metadata_path), "trace_path": str(trace_path)}
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                row["result"] = metadata.get("result")
                row["game_time_seconds"] = metadata.get("game_time_seconds")
                if trace_path.is_file():
                    decisions = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
                    row["decision_count"] = len(decisions)
                    row["laya_decision_count"] = sum(
                        item["decision"].get("executed_policy") == "laya" for item in decisions
                    )
                    row["model_revisions"] = sorted({
                        item["decision"]["model_revision"] for item in decisions
                        if item["decision"].get("model_revision")
                    })
                    row["executed_requested_policy_count"] = sum(
                        item["decision"].get("executed_policy") == policy for item in decisions
                    )
                    row["executed_policy_counts"] = {
                        name: sum(item["decision"].get("executed_policy") == name for item in decisions)
                        for name in ("rules", "laya")
                    }
                    laya_latencies = [
                        item["decision"]["latency_ms"] for item in decisions
                        if item["decision"].get("executed_policy") == "laya"
                        and isinstance(item["decision"].get("latency_ms"), (int, float))
                    ]
                    row["laya_latency_p50_ms"] = _percentile(laya_latencies, 0.50)
                    row["laya_latency_p95_ms"] = _percentile(laya_latencies, 0.95)
                    row["fallback_count"] = sum(
                        bool(item["decision"].get("fallback_reason")) for item in decisions
                    )
                    row["fallback_rate"] = row["fallback_count"] / len(decisions) if decisions else None
            row["complete"] = process.returncode == 0 and row.get("result") in {"Victory", "Defeat", "Tie"}
            row["strategy_verified"] = row["complete"] and row.get("executed_requested_policy_count", 0) > 0
            runs.append(row)
            pair["runs"].append(row)
        pairs.append(pair)

    summary = {}
    for policy in policies:
        policy_runs = [row for row in runs if row["policy"] == policy]
        completed = [row for row in policy_runs if row.get("complete")]
        finished = [row for row in completed if row.get("strategy_verified")]
        wins = sum(row.get("result") == "Victory" for row in finished)
        summary[policy] = {
            "completed_games": len(completed),
            "evaluated_games": len(finished),
            "scheduled_games": games_per_policy,
            "crashed_games": games_per_policy - len(completed),
            "crash_rate": (games_per_policy - len(completed)) / games_per_policy,
            "wins": wins,
            "losses": sum(row.get("result") == "Defeat" for row in finished),
            "ties": sum(row.get("result") == "Tie" for row in finished),
            "win_rate": wins / len(finished) if finished else None,
            "wilson_95": wilson_interval(wins, len(finished)),
            "mean_fallback_rate": _mean([row["fallback_rate"] for row in finished if row.get("fallback_rate") is not None]),
            "mean_laya_decisions": _mean([row["laya_decision_count"] for row in finished if "laya_decision_count" in row]),
            "mean_laya_latency_p50_ms": _mean([row["laya_latency_p50_ms"] for row in finished if row.get("laya_latency_p50_ms") is not None]),
            "mean_laya_latency_p95_ms": _mean([row["laya_latency_p95_ms"] for row in finished if row.get("laya_latency_p95_ms") is not None]),
        }

    report = {
        "schema_version": 1,
        "source": "yuri-paired-evaluation",
        "started_at": run_stamp,
        "map": map_name,
        "difficulty": difficulty,
        "realtime": realtime,
        "seed_start": seed_start,
        "games_per_policy": games_per_policy,
        "policies": list(policies),
        "status": "complete" if all(row.get("strategy_verified") for row in runs) else "incomplete",
        "summary": summary,
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _mean(values):
    return sum(values) / len(values) if values else None


def _percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * quantile
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
