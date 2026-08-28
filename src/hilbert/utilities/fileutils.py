"""
File and filesystem utility functions for Hilbert workflows.
"""

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def chdir(target_dir, create=False):
    """
    Context manager to temporarily change the working directory.

    :param target_dir: Target directory path to switch into.
    :param create: If True, create the directory if it does not exist.
    :return: Yields resolved Path object to the target directory.
    """
    prev_dir = Path.cwd()
    target = Path(target_dir).resolve()

    if create and not target.exists():
        target.mkdir(parents=True, exist_ok=True)

    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(prev_dir)
