"""

"utils/decorators.py"

Function decorators.

"""

import functools
import os
from timeit import default_timer as time


# ---------------- DECORATORS ----------------
def timer(message="Time elapsed"):
    """Timer decorator

    :param message: message to be pre-pended to time elapsed
    :returns: decorated function

    """

    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            print("{}: {}s".format(message, round(time() - start, 3)))
            return result

        return _func

    return _timer


def restore_cwd(func):
    """Decorator to restore `os.getcwd()` in functions that use `os.chdir`

    :param func: function
    :returns: decorated function

    """

    @functools.wraps(func)
    def _func(*args, **kwargs):
        cwd = os.getcwd()
        result = func(*args, **kwargs)
        os.chdir(cwd)
        return result

    return _func


def add_config(config_name, get_config, wraps=None):
    """Decorator that adds custom configuration to layer

    :param config_name: name of config dictionary (contained in CONFIGS)
    :param get_config: config retriever function
    :param wraps: function to wrap (default: None)
    :returns: decorated function

    """

    def _add_config(func):
        @functools.wraps(func if wraps is None else wraps)
        def _func(*args, **kwargs):
            config = get_config(config_name)
            config.update(kwargs)
            return func(*args, **config)

        return _func

    return _add_config