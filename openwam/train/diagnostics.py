"""Opt-in stack sampling without CPython's asynchronous frame traversal.

On the DLC Python 3.10.20 runtime, dump_traceback_later raced module/generator
frame traversal and reproduced SIGSEGV without CUDA. A Python sampler thread
calls dump_traceback while holding the GIL instead. This diagnoses Python and
CUDA waits that release the GIL; it cannot diagnose a permanently held GIL.
Use an external debugger for that case, and keep independent process deadlines.
"""
from contextlib import contextmanager
import faulthandler
import math
import threading


@contextmanager
def periodic_tracebacks(stream, *, interval=120.):
    """Sample until scope exit; join before the caller closes its output file."""
    if (isinstance(interval, bool) or not isinstance(interval, (int, float))
            or not math.isfinite(interval) or interval <= 0):
        raise ValueError('traceback interval must be finite and positive')
    stop = threading.Event()

    def sample():
        while not stop.wait(interval):
            faulthandler.dump_traceback(file=stream, all_threads=True)

    worker = threading.Thread(target=sample, name='openwam-stack-sampler', daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
