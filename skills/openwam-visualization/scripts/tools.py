#!/usr/bin/env python3
"""Discover existing renderers without importing them; plan or explicitly run one."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def find_repo(explicit=None):
    candidates = [Path(explicit)] if explicit else [Path(os.environ.get("OPENWAM_REPO", ".")),
                                                   Path.cwd(), Path(__file__).resolve()]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        for path in ([candidate] if explicit else [candidate, *candidate.parents]):
            if (path / "openwam/dataloader").is_dir() and (path / "scripts").is_dir():
                return path
    raise ValueError("OpenWAM checkout not found; pass --repo /path/to/OpenWAM")


def main():
    catalog = json.loads((Path(__file__).resolve().parents[1] / "references/tools.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--python", default=sys.executable, help="interpreter for the selected renderer")
    sub = parser.add_subparsers(dest="mode", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--check", action="store_true")
    for mode in ("plan", "run"):
        command = sub.add_parser(mode)
        command.add_argument("tool", choices=[x["id"] for x in catalog])
        command.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        repo = find_repo(args.repo)
        if args.mode == "list":
            rows = [dict(x, exists=(repo / x["path"]).is_file()) for x in catalog]
            print(json.dumps(dict(repo=str(repo), tools=rows), ensure_ascii=False, indent=2))
            return int(args.check and not all(x["exists"] for x in rows))
        tool = next(x for x in catalog if x["id"] == args.tool)
        if not (repo / tool["path"]).is_file():
            raise ValueError(f"Missing tool {tool['path']}; this skill needs its source checkout/cached drivers")
        extra = args.args[1:] if args.args[:1] == ["--"] else args.args
        if extra and not tool["flags"]:
            raise ValueError(f"{args.tool} has no CLI; do not pass invented flags or use --help to inspect it")
        unsupported = [x for x in extra if x.startswith("--") and x.split("=", 1)[0]
                       not in [*tool["flags"], "--help"]]
        if unsupported:
            raise ValueError(f"Unsupported flags {unsupported}; available: {tool['flags']}")
        entry = ["-m", tool["module"]] if "module" in tool else [str(repo / tool["path"])]
        command = [args.python, *entry, *extra]
        plan = dict(repo=str(repo), command=command, tool=tool,
                    note="Historical tools read dated caches; inspect/adapt them for the latest label contract."
                         if tool["historical"] else "Check inputs and output paths; some tools update a shared report.")
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.mode == "run":
            return subprocess.run(command, cwd=repo).returncode
        return 0
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
