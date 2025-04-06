from core.base import EpisodeEnded, EpisodeTimeout
import sys

def custom_excepthook(exctype, value, traceback):
    """Handles exceptions gracefully without stack traces."""
    if exctype in (EpisodeEnded, EpisodeTimeout):
        pass
    else:
        sys.__excepthook__(exctype, value, traceback)

def custom_thread_excepthook(args):
    """Handles exceptions in threads gracefully."""
    if isinstance(args.exc_value, (EpisodeEnded, EpisodeTimeout)):
        pass
    else:
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)
