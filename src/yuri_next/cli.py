import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
from pathlib import Path
import sys

from .contracts import Observation
from .policies import create_policy


def main():
    parser = argparse.ArgumentParser(description="Yuri policy laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    decide = commands.add_parser("decide", help="Evaluate an observation and export a replay")
    decide.add_argument("--policy", choices=("rules", "laya"), default="rules")
    decide.add_argument("--state", type=Path, required=True)
    decide.add_argument("--output", type=Path)
    decide.add_argument("--budget-ms", type=float, default=1500)
    decide.add_argument("--min-confidence", type=float, default=0.65)
    decide.add_argument("--revision", default=os.environ.get("YURI_LAYA_REVISION"))
    preflight = commands.add_parser("preflight", help="Run real Laya inference and report runtime/model details")
    preflight.add_argument("--state", type=Path, required=True)
    preflight.add_argument("--output", type=Path)
    preflight.add_argument("--budget-ms", type=float, default=120000)
    preflight.add_argument("--revision", default=os.environ.get("YURI_LAYA_REVISION"))
    game = commands.add_parser("game", help="Launch the modern experimental SC2 bot")
    game.add_argument("--policy", choices=("rules", "laya"), default="rules")
    game.add_argument("--map", default="AcropolisLE")
    game.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    game.add_argument("--seed", type=int, default=1)
    game.add_argument("--run-id")
    game.add_argument("--realtime", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Run reproducible paired matches and write a report")
    evaluate.add_argument("--policies", nargs="+", choices=("rules", "laya"), default=("rules", "laya"))
    evaluate.add_argument("--games-per-policy", type=int, default=10)
    evaluate.add_argument("--seed-start", type=int, default=1)
    evaluate.add_argument("--map", default="AcropolisLE")
    evaluate.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    evaluate.add_argument("--realtime", action="store_true")
    evaluate.add_argument("--output", type=Path)
    doctor = commands.add_parser("doctor", help="Inspect SC2 installation, map, and WSL2 connection settings")
    doctor.add_argument("--map", default="AcropolisLE")
    args = parser.parse_args()
    if args.command == "doctor":
        from sc2 import maps
        from sc2.paths import PF, Paths
        sc2_path = None
        executable = None
        install_error = None
        try:
            sc2_path = Paths.BASE
            executable = Paths.EXECUTABLE
        except SystemExit as error:
            install_error = str(error) or "SC2 installation could not be resolved"
        try:
            selected_map = maps.get(args.map)
            map_check = {"available": True, "name": selected_map.name,
                         "path": str(selected_map.path)}
        except (KeyError, FileNotFoundError, SystemExit) as error:
            map_check = {"available": False, "name": args.map, "error": str(error)}
        wsl2 = PF == "WSL2"
        checks = {
            "platform": PF,
            "sc2_path": str(sc2_path or os.environ.get("SC2PATH") or ""),
            "executable": {"available": bool(executable and executable.is_file()),
                           "path": str(executable) if executable else None,
                           "error": install_error},
            "map": map_check,
            "wsl_connection": {
                "required": wsl2,
                "clienthost_set": bool(os.environ.get("SC2CLIENTHOST")),
                "serverhost_set": bool(os.environ.get("SC2SERVERHOST")),
                "clienthost": os.environ.get("SC2CLIENTHOST"),
                "serverhost": os.environ.get("SC2SERVERHOST"),
            },
        }
        passed = checks["executable"]["available"] and map_check["available"]
        if wsl2:
            passed = passed and checks["wsl_connection"]["clienthost_set"] and checks["wsl_connection"]["serverhost_set"]
        print(json.dumps({"status": "passed" if passed else "failed", "checks": checks},
                         ensure_ascii=False, indent=2))
        if not passed:
            sys.exit(1)
        return
    if args.command == "game":
        from .game import launch
        launch(args.policy, args.map, args.difficulty, args.realtime, args.seed, args.run_id)
        return
    if args.command == "evaluate":
        from datetime import datetime, timezone
        from .evaluation import run_paired_evaluation
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output or Path("artifacts/evaluations") / f"{stamp}.json"
        report = run_paired_evaluation(
            policies=args.policies, games_per_policy=args.games_per_policy,
            seed_start=args.seed_start, map_name=args.map, difficulty=args.difficulty,
            realtime=args.realtime, output=output,
        )
        print(json.dumps({"status": report["status"], "report": str(output),
                          "summary": report["summary"]}, ensure_ascii=False, indent=2))
        if report["status"] != "complete":
            sys.exit(1)
        return
    observation = Observation(**json.loads(args.state.read_text(encoding="utf-8")))
    policy_name = "laya" if args.command == "preflight" else args.policy
    kwargs = {}
    if policy_name == "laya":
        kwargs = {"budget_ms": args.budget_ms, "revision": args.revision}
        if args.command == "preflight":
            kwargs["min_confidence"] = 0.0
        else:
            kwargs["min_confidence"] = args.min_confidence
    decision = create_policy(policy_name, **kwargs).decide(observation)
    record = {"schema_version": 1, "source": "python-runtime",
              "observation": observation.to_dict(), "decision": decision.to_dict()}
    if args.command == "preflight":
        try:
            laya_version = version("laya")
        except PackageNotFoundError:
            laya_version = None
        runtime = {"python": platform.python_version(), "platform": platform.platform(),
                   "laya": laya_version}
        try:
            import torch
            runtime["torch"] = torch.__version__
            runtime["cuda"] = torch.version.cuda
            runtime["cuda_available"] = torch.cuda.is_available()
            runtime["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            runtime["device_memory_bytes"] = (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available() else None
            )
        except Exception as error:
            runtime["torch_error"] = f"{type(error).__name__}: {error}"
        record["source"] = "laya-preflight"
        record["runtime"] = runtime
        preflight_passed = decision.executed_policy == "laya" and bool(decision.model_revision)
        record["status"] = "passed" if preflight_passed else "failed"
        if decision.executed_policy != "laya":
            record["status_reason"] = decision.fallback_reason or "laya_inference_not_executed"
        elif not decision.model_revision:
            record["status_reason"] = "model_revision_unavailable"
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.command == "preflight" and record["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
