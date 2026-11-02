#!/usr/bin/env python3
"""Run a bounded Stage1 benchmark with GIL-held periodic stack diagnostics.

The optional --benchmark-script pins an existing immutable benchmark checkout;
its own prepared-runtime validation remains authoritative and unmodified.
All remaining arguments are passed to benchmark_stage1.py unchanged.
"""
import argparse
import faulthandler
import importlib.util
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--trace-interval-seconds', type=float, default=120.)
    parser.add_argument('--benchmark-script', type=Path,
                        default=ROOT / 'scripts/benchmark_stage1.py')
    args, remaining = parser.parse_known_args(argv)
    if '--output-dir' not in remaining:
        parser.error('benchmark --output-dir is required')
    output = Path(remaining[remaining.index('--output-dir') + 1]) / 'control'
    output.mkdir(parents=True, exist_ok=True)
    # Do not import this checkout's openwam package before the selected entry:
    # an explicitly pinned benchmark must resolve its own training modules and
    # prepared-runtime fingerprint, even when this wrapper lives elsewhere.
    spec = importlib.util.spec_from_file_location(
        '_openwam_stack_diagnostics', ROOT / 'openwam/train/diagnostics.py')
    diagnostics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics)
    previous = sys.argv
    try:
        with (output / ('stack_rank_' + os.environ.get('RANK', '0') + '.txt')).open('a') as stream:
            with diagnostics.periodic_tracebacks(stream, interval=args.trace_interval_seconds):
                # Fatal signals stay on stderr; no handler retains the stream
                # after this scope closes it.
                faulthandler.enable()
                sys.argv = [str(args.benchmark_script), *remaining]
                runpy.run_path(str(args.benchmark_script), run_name='__main__')
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
