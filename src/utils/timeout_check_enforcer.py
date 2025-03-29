from functools import wraps
import logging as log

def enforce_timeout_check(method):
    """
    Decorator to automatically call `check_timeout()` before executing the packet sending logic.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        log.info("apero")
        self.check_timeout()
        return method(self, *args, **kwargs)
    return wrapper
