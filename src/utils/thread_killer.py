import ctypes
import time
import threading
from contextlib import contextmanager
import os
import sys
import inspect

@contextmanager
def silence_stderr():
    """Context manager to silence stderr."""
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr = old_stderr

def kill_thread(thread: threading.Thread, max_attempts: int = 50, escalation_delay: float = 0.1):
    """
    Intenta matar un hilo con múltiples estrategias de intensidad creciente.

    Args:
        thread: El hilo a terminar
        max_attempts: Número máximo de intentos por estrategia
        escalation_delay: Tiempo entre intentos
    """
    if not thread.is_alive():
        return True

    # Estrategias de terminación en orden de intensidad
    strategies = [
        _try_soft_termination,
        _try_system_exit,
        _try_keyboard_interrupt,
        _try_async_raise,
        _try_thread_abort
    ]

    for strategy in strategies:
        for attempt in range(max_attempts):
            if not thread.is_alive():
                return True

            try:
                with silence_stderr():
                    if strategy(thread):
                        time.sleep(escalation_delay)
                        if not thread.is_alive():
                            return True
            except Exception as e:
                log.warning(f"Strategy {strategy.__name__} failed: {str(e)}")
                continue

    # Último recurso: marcar como daemon si no es crítico
    if thread.is_alive():
        thread.daemon = True
        log.error("Failed to kill thread, marking as daemon. This may cause resource leaks!")
        return False
    return True

def _try_soft_termination(thread):
    """Intenta terminación suave mediante flags"""
    if hasattr(thread, '_stop_event'):
        thread._stop_event.set()
        return True
    return False

def _try_system_exit(thread):
    """Intenta con SystemExit"""
    tid = ctypes.c_long(thread.ident)
    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, exc)
    return res == 1

def _try_keyboard_interrupt(thread):
    """Intenta con KeyboardInterrupt (más difícil de ignorar)"""
    tid = ctypes.c_long(thread.ident)
    exc = ctypes.py_object(KeyboardInterrupt)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, exc)
    return res == 1

def _try_async_raise(thread):
    """Versión más agresiva de levantamiento de excepciones"""
    if not inspect.isclass(SystemExit):
        return False

    tid = ctypes.c_long(thread.ident)
    exc = ctypes.py_object(SystemExit)

    # Intenta forzar la excepción múltiples veces
    success = False
    for _ in range(3):
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, exc)
        if res == 1:
            success = True
        time.sleep(0.01)

    return success

def _try_thread_abort(thread):
    """Estrategia nuclear - puede causar inestabilidad"""
    # Solo usar como último recurso
    if not hasattr(ctypes.pythonapi, 'PyThreadState_SetAsyncExc'):
        return False

    tid = ctypes.c_long(thread.ident)
    exc = ctypes.py_object(BaseException)  # La excepción más genérica posible

    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, exc)
    if res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)

    return res == 1
