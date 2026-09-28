import argparse
import json
from pathlib import Path

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
    game = commands.add_parser("game", help="Launch the modern experimental SC2 bot")
    game.add_argument("--policy", choices=("rules", "laya"), default="rules")
    game.add_argument("--map", default="AbyssalReefLE")
    game.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    game.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    if args.command == "game":
        from .game import launch
        launch(args.policy, args.map, args.difficulty, args.realtime)
        return
    observation = Observation(**json.loads(args.state.read_text(encoding="utf-8")))
    kwargs = {"budget_ms": args.budget_ms} if args.policy == "laya" else {}
    decision = create_policy(args.policy, **kwargs).decide(observation)
    record = {"schema_version": 1, "source": "python-runtime",
              "observation": observation.to_dict(), "decision": decision.to_dict()}
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
