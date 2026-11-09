"""Periodic diagnostics must not race CPython's frame traversal without the GIL."""
import faulthandler
import threading
import time

import pytest


def test_periodic_dump_uses_python_thread_and_stops_before_file_close(tmp_path, monkeypatch):
    from openwam.train.diagnostics import periodic_tracebacks

    def forbidden(*args, **kwargs):
        raise AssertionError('native asynchronous frame traversal caused DLC SIGSEGV')

    monkeypatch.setattr(faulthandler, 'dump_traceback_later', forbidden)
    calls = []
    sampled = threading.Event()
    original = faulthandler.dump_traceback

    def record(**kwargs):
        calls.append(threading.current_thread())
        original(**kwargs)
        sampled.set()

    monkeypatch.setattr(faulthandler, 'dump_traceback', record)
    path = tmp_path / 'stacks.txt'
    with path.open('w') as stream:
        with periodic_tracebacks(stream, interval=.005):
            assert sampled.wait(3)
        assert calls and all(not thread.is_alive() for thread in calls)
    count = len(calls)
    time.sleep(.02)
    assert len(calls) == count
    assert 'test_periodic_dump_uses_python_thread' in path.read_text()


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf'), True])
def test_invalid_interval_fails_before_starting_sampler(tmp_path, interval):
    from openwam.train.diagnostics import periodic_tracebacks
    with (tmp_path / 'stacks.txt').open('w') as stream:
        with pytest.raises(ValueError, match='interval'):
            with periodic_tracebacks(stream, interval=interval):
                raise AssertionError('invalid interval accepted')


def test_sampler_joins_on_training_exception(tmp_path):
    from openwam.train.diagnostics import periodic_tracebacks
    before = set(threading.enumerate())
    with (tmp_path / 'stacks.txt').open('w') as stream:
        with pytest.raises(RuntimeError, match='training failed'):
            with periodic_tracebacks(stream, interval=100):
                raise RuntimeError('training failed')
    assert set(threading.enumerate()) == before


def test_trace_entry_preserves_pinned_benchmark_imports_and_arguments(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    entry = tmp_path / 'pinned_benchmark.py'
    entry.write_text('''import sys,json,time
assert 'openwam' not in sys.modules, 'wrapper imported a different training checkout'
print(json.dumps(sys.argv[1:]),flush=True)
time.sleep(.05)
''')
    args = ['--output-dir', str(tmp_path / 'out'), '--case', 'untouched']
    script = Path(__file__).resolve().parents[2] / 'scripts/experiments/trace_stage1_benchmark.py'
    result = subprocess.run([sys.executable, str(script), '--trace-interval-seconds', '.005',
                             '--benchmark-script', str(entry), *args],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == args
    assert 'pinned_benchmark.py' in (tmp_path / 'out/control/stack_rank_0.txt').read_text()
