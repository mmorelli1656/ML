# -*- coding: utf-8 -*-

"""
submodules_paths
=============

Utility functions to add Git submodules to the Python module search path.

This module allows you to import code from submodules (e.g., ``Utils``, 
``ML_Submodule``) without having to hardcode absolute paths.

Functions
---------
add_submodule_to_path(submodule)
    Add a single submodule directory to the Python module search path.
add_submodules_to_path(submodules)
    Add multiple submodule directories to the Python module search path.

Examples
--------
Add a single submodule:

>>> from project_paths import add_submodule_to_path
>>> add_submodule_to_path("Utils")
'Added to sys.path: C:/Users/username/Github/project/Utils'

Add multiple submodules at once:

>>> from project_paths import add_submodules_to_path
>>> add_submodules_to_path(["Utils", "ML_Submodule"])
['Added to sys.path: .../Utils', 'Added to sys.path: .../ML_Submodule']
"""

import sys
from pathlib import Path


def add_submodule_to_path(submodule: str) -> str:
    """
    Add a single submodule directory to the Python module search path (`sys.path`).

    Parameters
    ----------
    submodule : str
        Name of the submodule folder to add.

    Returns
    -------
    str
        Confirmation message indicating the path that was added to `sys.path`.

    Raises
    ------
    FileNotFoundError
        If the specified submodule directory does not exist.
    """
    project_root = Path(__file__).resolve().parent
    sub_path = project_root / submodule

    if sub_path.exists():
        if str(sub_path) not in sys.path:  # avoid duplicates
            sys.path.append(str(sub_path))
        return f"Added to sys.path: {sub_path}"
    else:
        raise FileNotFoundError(f"The path {sub_path} does not exist.")


def add_submodules_to_path(submodules: list[str]) -> list[str]:
    """
    Add multiple submodule directories to the Python module search path (`sys.path`).

    Parameters
    ----------
    submodules : list of str
        Names of the submodule folders to add.

    Returns
    -------
    list of str
        Confirmation messages for each successfully added submodule.

    Raises
    ------
    FileNotFoundError
        If one or more submodule directories do not exist.
    """
    messages = []
    for submodule in submodules:
        messages.append(add_submodule_to_path(submodule))
    return messages


# Example usage when running directly
if __name__ == "__main__":
    print(add_submodule_to_path("Utils"))
    print(add_submodule_to_path("ML_Submodule"))
    # OR
    # print(add_submodules_to_path(["Utils", "ML_Submodule"]))
