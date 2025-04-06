import ctypes
import threading
import sys
import os
import contextlib

@contextlib.contextmanager
def silence_stderr():
    """Context manager to silence stderr."""
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr = old_stderr

def kill_thread(thread: threading.Thread):
    """Kills a thread forcefully using ctypes and silences the traceback."""
    if not thread.is_alive():
        return

    tid = ctypes.c_long(thread.ident)
    
    # Silenciamos stderr mientras ejecutamos la terminación del hilo
    with silence_stderr():
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))

        if res == 0:
            raise ValueError(f"Invalid thread ID: {tid.value}")
        elif res != 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError(f"Failed to terminate thread {tid.value} properly.")

        try:
            thread.join(timeout=0.1)
        except SystemExit:
            pass
