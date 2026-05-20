from __future__ import annotations

import argparse
from pathlib import Path

from .config import default_config, default_machine, load_config, load_run_config, write_yaml
from .demo import init_demo
from .dft import collect_vasp, prepare_vasp, submit_vasp, wait_vasp
from .explore import run_explore
from .mace import foundation_models, init_committee_from_foundations, train_committee
from .paths import Layout
from .select import run_select
from .workflow import run_loop
from .report import report
from .runner import export_final_models, run_automatic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MACE active-learning workflow")
    parser.add_argument("--config", default="job.yaml")
    parser.add_argument("--root", default=".")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("init-dpgen")
    p.add_argument("--param", default="param.json")
    p.add_argument("--machine", default="machine.json")
    sub.add_parser("check")
    p = sub.add_parser("init-demo")
    p.add_argument("--source", default=None)
    p = sub.add_parser("report")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("init-committee")
    p.add_argument("-g", "--generation", type=int, default=0)
    p.add_argument("--copy", action="store_true", help="Copy foundation models instead of symlinking")
    for name in ["mace", "explore", "select", "prepare-vasp", "submit-vasp", "wait-vasp", "collect"]:
        p = sub.add_parser(name)
        p.add_argument("-g", "--generation", type=int, default=None)
    p = sub.add_parser("train")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--wait", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("param")
    p.add_argument("machine")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p = sub.add_parser("run-report")
    p.add_argument("param")
    p.add_argument("machine")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("param")
    p.add_argument("machine")
    return parser


def load_with_layout(args) -> tuple[dict, Layout]:
    root = Path(args.root).resolve()
    cfg = load_config(root / args.config)
    generation = getattr(args, "generation", None)
    layout = Layout.from_config(cfg, root=root, generation=generation)
    return cfg, layout


def check(cfg: dict, layout: Layout) -> None:
    print(f"root: {layout.root}")
    print(f"work_path: {layout.work_path}")
    for label, value in [
        ("seed_file", cfg["project"]["seed_file"]),
        ("train_file", cfg["project"]["train_file"]),
        ("foundation_model", cfg["project"]["foundation_model"]),
        ("template_dir", cfg["dft"]["template_dir"]),
    ]:
        path = layout.rel(value)
        print(f"{label}: {path} {'OK' if path.exists() else 'MISSING'}")
    for module in ["ase", "mace", "yaml"]:
        try:
            __import__(module)
            print(f"{module}: OK")
        except Exception as exc:
            print(f"{module}: MISSING ({exc})")
    try:
        models = foundation_models(cfg, layout)
        print("foundation_models:")
        for model in models:
            print(f"  {model} OK")
    except Exception as exc:
        print(f"foundation_models: MISSING ({exc})")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.cmd == "init":
        write_yaml(root / args.config, default_config())
        print(root / args.config)
        return
    if args.cmd == "init-dpgen":
        import json

        (root / args.param).write_text(json.dumps(default_config(), indent=2) + "\n", encoding="utf-8")
        (root / args.machine).write_text(json.dumps(default_machine(), indent=2) + "\n", encoding="utf-8")
        print(root / args.param)
        print(root / args.machine)
        return
    if args.cmd == "init-demo":
        init_demo(root, args.source) if args.source else init_demo(root)
        return
    if args.cmd == "run":
        cfg, run_root = load_run_config(args.param, args.machine)
        run_automatic(cfg, run_root, start=args.start, stop=args.stop)
        return
    if args.cmd == "run-report":
        cfg, run_root = load_run_config(args.param, args.machine)
        report(cfg, run_root, args.verbose)
        return
    if args.cmd == "export":
        cfg, run_root = load_run_config(args.param, args.machine)
        export_final_models(cfg, run_root)
        return

    cfg, layout = load_with_layout(args)
    if args.cmd == "check":
        check(cfg, layout)
    elif args.cmd == "init-committee":
        print(init_committee_from_foundations(cfg, layout, copy=args.copy))
    elif args.cmd == "mace":
        print(train_committee(cfg, layout))
    elif args.cmd == "explore":
        print(run_explore(cfg, layout))
    elif args.cmd == "select":
        print(run_select(cfg, layout))
    elif args.cmd == "prepare-vasp":
        print(prepare_vasp(cfg, layout))
    elif args.cmd == "submit-vasp":
        submit_vasp(cfg, layout)
    elif args.cmd == "wait-vasp":
        wait_vasp(cfg, layout)
    elif args.cmd == "collect":
        print(collect_vasp(cfg, layout))
    elif args.cmd == "report":
        report(cfg, root, args.verbose)
    elif args.cmd == "train":
        run_loop(cfg, root, args.start, args.stop, args.submit, args.wait)


if __name__ == "__main__":
    main()
