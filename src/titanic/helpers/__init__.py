"""Helper functions and classes for the titanic project.

This package provides utility functions and classes that are used throughout
the project. These helpers are designed to be reusable and modular, making
it easy to incorporate them into different parts of the project.

The main functionalities include:

- **Package checking**: Verify if Python packages are installed
- **Function inspection**: Check function signatures and arguments
- **Compatibility utilities**: Provide compatibility shims for optional dependencies

Examples
--------
Check if a package is installed:

    >>> from titanic.helpers import is_installed
    >>> if is_installed("numpy"):
    ...     import numpy as np

Check if a function accepts a specific argument:

    >>> from titanic.helpers import is_argument
    >>> def my_func(x, y):
    ...     return x + y
    >>> is_argument(my_func, "x")  # True
    >>> is_argument(my_func, "z")  # False

Use a dummy progress bar when tqdm is not available:

    >>> from titanic.helpers import dummy_tqdm
    >>> for item in dummy_tqdm([1, 2, 3]):
    ...     process(item)

Notes
-----
This package is primarily intended for internal use within the titanic
project, but the functions are designed to be reusable in other projects as
well.

See Also
--------
- `titanic.logging` : Logging utilities

"""

from ._helpers import dummy_tqdm, is_argument, is_installed

# _helpers.pyだと、_が入っているのでドキュメント化されない。
# ドキュメント化したい場合は、モジュールメソッドとして登録するため、__all__に入れる。
__all__ = ("dummy_tqdm", "is_argument", "is_installed")
