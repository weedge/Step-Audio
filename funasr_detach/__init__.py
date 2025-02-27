"""Initialize funasr package."""

import logging
import os
import pkgutil
import importlib

try:
    from funasr_detach import version
    __version__ = version.__version__
except Exception as e:
    __version__ = "1.0.8"
    logging.warning(f"set __version__ to {__version__} because {e}")


import importlib
import pkgutil


def import_submodules(package, recursive=True):
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            results[name] = importlib.import_module(name)
        except Exception as e:
            # 如果想要看到导入错误的具体信息，可以取消注释下面的行
            # print(f"Failed to import {name}: {e}")
            pass
        if recursive and is_pkg:
            results.update(import_submodules(name))
    return results


import_submodules(__name__)

from funasr_detach.auto.auto_model import AutoModel
from funasr_detach.auto.auto_frontend import AutoFrontend
