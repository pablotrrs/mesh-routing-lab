from functools import wraps

def enforce_timeout_check(method):
    """
    Decorator to automatically call `check_timeout()` before executing the packet sending logic.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.check_timeout()
        return method(self, *args, **kwargs)
    return wrapper
